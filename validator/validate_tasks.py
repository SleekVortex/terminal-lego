#!/usr/bin/env python3
"""
Docker Round-Trip Validator for Terminal-Lego Tasks

For each candidate task:
1. docker build the image from environment/Dockerfile
2. Start container, copy in solution/solve.sh, execute it
3. Copy in tests/ directory, execute tests/test.sh
4. Read /logs/verifier/reward.txt
5. reward=1 → copy to output dir; otherwise skip

Usage:
    python validate_tasks.py --input ./candidates --output ./validated --workers 8 --timeout 300
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('validate_tasks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AtomicCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value


progress = AtomicCounter()
passed = AtomicCounter()
failed = AtomicCounter()
build_failed = AtomicCounter()
error_count = AtomicCounter()
def run_cmd(cmd: list, timeout: int = 300, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)


def validate_task(task_dir: Path, output_dir: Path, timeout: int, total: int) -> dict:
    task_name = task_dir.name
    image_tag = f"tl-validate-{task_name}".lower()
    container_name = f"tl-val-{task_name}-{os.getpid()}".lower()
    result = {
        "task": task_name,
        "status": "unknown",
        "reward": None,
        "build_time": 0,
        "solve_time": 0,
        "test_time": 0,
        "error": None,
    }

    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"
    solve_sh = task_dir / "solution" / "solve.sh"
    tests_dir = task_dir / "tests"
    test_sh = tests_dir / "test.sh"

    if not dockerfile.exists():
        result["status"] = "missing_dockerfile"
        result["error"] = "No Dockerfile found"
        return result
    if not solve_sh.exists():
        result["status"] = "missing_solution"
        result["error"] = "No solve.sh found"
        return result
    if not test_sh.exists():
        result["status"] = "missing_tests"
        result["error"] = "No test.sh found"
        return result

    try:
        t0 = time.time()
        build_result = run_cmd(
            ["docker", "build", "-t", image_tag, "-f", str(dockerfile), str(env_dir)],
            timeout=timeout
        )
        result["build_time"] = round(time.time() - t0, 1)

        if build_result.returncode != 0:
            result["status"] = "build_failed"
            result["error"] = build_result.stderr[-500:] if build_result.stderr else "build failed"
            build_failed.increment()
            return result

        run_result = run_cmd(
            ["docker", "run", "-d",
             "--name", container_name,
             "--memory", "1g",
             "--cpus", "1",
             image_tag,
             "sleep", "infinity"],
            timeout=30
        )
        if run_result.returncode != 0:
            result["status"] = "container_start_failed"
            result["error"] = run_result.stderr[-500:] if run_result.stderr else "container start failed"
            return result

        run_cmd(["docker", "exec", container_name, "mkdir", "-p", "/logs/verifier"], timeout=10)
        run_cmd(["docker", "exec", container_name, "mkdir", "-p", "/tests"], timeout=10)

        run_cmd(["docker", "cp", str(solve_sh), f"{container_name}:/app/solve.sh"], timeout=10)
        run_cmd(["docker", "exec", container_name, "chmod", "+x", "/app/solve.sh"], timeout=10)

        t1 = time.time()
        solve_result = run_cmd(
            ["docker", "exec", "-w", "/app", container_name, "bash", "/app/solve.sh"],
            timeout=timeout
        )
        result["solve_time"] = round(time.time() - t1, 1)

        run_cmd(["docker", "cp", str(tests_dir) + "/.", f"{container_name}:/tests/"], timeout=10)
        run_cmd(["docker", "exec", container_name, "chmod", "+x", "/tests/test.sh"], timeout=10)

        t2 = time.time()
        run_cmd(
            ["docker", "exec", "-w", "/app", container_name, "bash", "/tests/test.sh"],
            timeout=timeout
        )
        result["test_time"] = round(time.time() - t2, 1)

        reward_result = run_cmd(
            ["docker", "exec", container_name, "cat", "/logs/verifier/reward.txt"],
            timeout=10
        )

        if reward_result.returncode == 0:
            try:
                result["reward"] = float(reward_result.stdout.strip())
            except ValueError:
                result["reward"] = 0
        else:
            result["reward"] = 0

        if result["reward"] == 1:
            result["status"] = "passed"
            passed.increment()
            dest = output_dir / task_name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(task_dir, dest)
            logger.info(f"[{task_name}] PASSED (build={result['build_time']}s solve={result['solve_time']}s test={result['test_time']}s)")
        else:
            result["status"] = "failed"
            failed.increment()
            logger.info(f"[{task_name}] FAILED reward={result['reward']}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"Timeout after {timeout}s"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        error_count.increment()
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
        except Exception:
            pass
        try:
            subprocess.run(["docker", "rmi", "-f", image_tag], capture_output=True, timeout=30)
        except Exception:
            pass

        current = progress.increment()
        if current % 10 == 0 or current == total:
            logger.info(f"Progress: {current}/{total} | passed={passed.value} failed={failed.value}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Docker Round-Trip Validator')
    parser.add_argument('--input', '-i', required=True, help='Input directory with candidate tasks')
    parser.add_argument('--output', '-o', required=True, help='Output directory for validated tasks')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Parallel workers')
    parser.add_argument('--timeout', '-t', type=int, default=300, help='Timeout per step (seconds)')
    parser.add_argument('--limit', '-l', type=int, default=None, help='Limit tasks to validate')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("task_")
    ])

    if args.limit:
        task_dirs = task_dirs[:args.limit]

    total = len(task_dirs)
    logger.info(f"Found {total} tasks to validate in {input_dir}")
    logger.info(f"Output: {output_dir}, Workers: {args.workers}, Timeout: {args.timeout}s")

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate_task, td, output_dir, args.timeout, total): td
            for td in task_dirs
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                task_dir = futures[future]
                logger.exception(f"[{task_dir.name}] Unhandled exception: {e}")
                results.append({"task": task_dir.name, "status": "error", "error": str(e)})

    elapsed = time.time() - start_time

    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    logger.info("=" * 60)
    logger.info(f"Validation Complete! Time: {elapsed:.1f}s, Total: {total}")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status}: {count}")
    if total > 0:
        logger.info(f"Pass rate: {passed.value}/{total} ({passed.value/total*100:.1f}%)")
    logger.info("=" * 60)

    report = {
        "total": total,
        "passed": passed.value,
        "failed": failed.value,
        "build_failed": build_failed.value,
        "errors": error_count.value,
        "status_counts": status_counts,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
