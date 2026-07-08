#!/usr/bin/env python3
"""Docker round-trip validator for Terminal-Lego tasks."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from validator.docker_runner import DockerRunner
from validator.validation_logs import TaskLogWriter


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def add_output_file_logger(log_path: Path) -> None:
    root_logger = logging.getLogger()
    resolved_log_path = log_path.resolve()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved_log_path:
            return
    file_handler = logging.FileHandler(resolved_log_path)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)


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


class TaskValidator:
    def __init__(self, output_dir: Path, timeout: int, emit_progress: bool = True):
        self.output_dir = output_dir
        self.timeout = timeout
        self.emit_progress = emit_progress

    def validate(self, task_dir: Path, total: int) -> dict:
        task_name = task_dir.name
        runner = DockerRunner(task_name)
        logs = TaskLogWriter(self.output_dir, task_name)
        result = {
            "task": task_name,
            "status": "unknown",
            "reward": None,
            "build_time": 0,
            "solve_time": 0,
            "test_time": 0,
            "error": None,
            "log_dir": logs.relative_to(self.output_dir),
        }

        env_dir = task_dir / "environment"
        dockerfile = env_dir / "Dockerfile"
        solve_sh = task_dir / "solution" / "solve.sh"
        tests_dir = task_dir / "tests"
        test_sh = tests_dir / "test.sh"

        try:
            if not dockerfile.exists():
                result.update(status="missing_dockerfile", error="No Dockerfile found")
                return result
            if not solve_sh.exists():
                result.update(status="missing_solution", error="No solve.sh found")
                return result
            if not test_sh.exists():
                result.update(status="missing_tests", error="No test.sh found")
                return result

            t0 = time.time()
            build_result = runner.build(dockerfile, env_dir, self.timeout)
            result["build_time"] = round(time.time() - t0, 1)
            logs.write_command("build", build_result)
            if build_result.returncode != 0:
                result.update(status="build_failed", error=(build_result.stderr or build_result.stdout or "build failed")[-500:])
                build_failed.increment()
                return result

            run_result = runner.start()
            logs.write_command("run", run_result)
            if run_result.returncode != 0:
                result.update(status="container_start_failed", error=(run_result.stderr or run_result.stdout or "container start failed")[-500:])
                return result

            for stage, command in (
                ("mkdir_logs", ["mkdir", "-p", "/logs/verifier"]),
                ("mkdir_tests", ["mkdir", "-p", "/tests"]),
            ):
                logs.write_command(stage, runner.exec(command, timeout=10))

            logs.write_command("copy_solution", runner.copy_to_container(str(solve_sh), "/app/solve.sh", timeout=10))
            logs.write_command("chmod_solution", runner.exec(["chmod", "+x", "/app/solve.sh"], timeout=10))

            t1 = time.time()
            solve_result = runner.exec(["bash", "/app/solve.sh"], timeout=self.timeout, workdir="/app")
            result["solve_time"] = round(time.time() - t1, 1)
            logs.write_command("solve", solve_result)

            logs.write_command("copy_tests", runner.copy_to_container(str(tests_dir) + "/.", "/tests/", timeout=10))
            logs.write_command("chmod_tests", runner.exec(["chmod", "+x", "/tests/test.sh"], timeout=10))

            t2 = time.time()
            test_result = runner.exec(["bash", "/tests/test.sh"], timeout=self.timeout, workdir="/app")
            result["test_time"] = round(time.time() - t2, 1)
            logs.write_command("test", test_result)

            reward_result = runner.read_reward()
            logs.write_command("reward", reward_result)
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
                dest = self.output_dir / task_name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(task_dir, dest)
                logger.info(
                    "[%s] PASSED (build=%ss solve=%ss test=%ss)",
                    task_name,
                    result["build_time"],
                    result["solve_time"],
                    result["test_time"],
                )
            else:
                result["status"] = "failed"
                failed.increment()
                logger.info("[%s] FAILED reward=%s", task_name, result["reward"])
            return result
        except Exception as exc:
            result.update(status="error", error=str(exc))
            error_count.increment()
            return result
        finally:
            try:
                logs.write_result(result)
            except Exception:
                pass
            try:
                runner.cleanup()
            except Exception:
                pass
            if self.emit_progress:
                current = progress.increment()
                if current % 10 == 0 or current == total:
                    logger.info("Progress: %s/%s | passed=%s failed=%s", current, total, passed.value, failed.value)


def validate_task(task_dir: Path, output_dir: Path, timeout: int, total: int) -> dict:
    return TaskValidator(output_dir, timeout).validate(task_dir, total)


def load_task_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    tasks: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            tasks.add(line)
            continue
        if isinstance(row, dict):
            task_name = row.get("task") or row.get("task_name")
            if task_name:
                tasks.add(str(task_name))
        elif isinstance(row, str):
            tasks.add(row)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Docker Round-Trip Validator")
    parser.add_argument("--input", "-i", required=True, help="Input directory with candidate tasks")
    parser.add_argument("--output", "-o", required=True, help="Output directory for validated tasks")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Parallel workers")
    parser.add_argument("--timeout", "-t", type=int, default=300, help="Timeout per step (seconds)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit tasks to validate")
    parser.add_argument("--task-list", type=Path, default=None, help="JSONL or text file with task names to validate.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    add_output_file_logger(output_dir / "validate_tasks.log")

    task_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
    task_filter = load_task_filter(args.task_list)
    if task_filter is not None:
        task_dirs = [task_dir for task_dir in task_dirs if task_dir.name in task_filter]
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    total = len(task_dirs)
    logger.info("Found %s tasks to validate in %s", total, input_dir)
    logger.info("Output: %s, Workers: %s, Timeout: %ss", output_dir, args.workers, args.timeout)

    start_time = time.time()
    results = []
    validator = TaskValidator(output_dir, args.timeout)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(validator.validate, task_dir, total): task_dir for task_dir in task_dirs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                task_dir = futures[future]
                logger.exception("[%s] Unhandled exception: %s", task_dir.name, exc)
                results.append({"task": task_dir.name, "status": "error", "error": str(exc)})

    elapsed = time.time() - start_time
    status_counts: dict[str, int] = {}
    for row in results:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    logger.info("=" * 60)
    logger.info("Validation Complete! Time: %.1fs, Total: %s", elapsed, total)
    for status, count in sorted(status_counts.items()):
        logger.info("  %s: %s", status, count)
    if total > 0:
        logger.info("Pass rate: %s/%s (%.1f%%)", passed.value, total, passed.value / total * 100)
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
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
