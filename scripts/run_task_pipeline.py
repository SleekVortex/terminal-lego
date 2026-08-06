#!/usr/bin/env python3
"""Run the simple production Terminal-Lego task pipeline.

Production policy:
1. prepare StackOverflow seed
2. generate task candidates once
3. submit each completed candidate immediately to Docker validation
4. keep only tasks whose golden solution receives reward == 1

Failed tasks are logged by the validator and skipped. This entrypoint
intentionally does not run Dockerfile regeneration or repair loops.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from generator import task_generator
from generator.resume import is_complete_task_dir
from validator.validate_tasks import (
    TaskValidator,
    add_output_file_logger,
    finalize_validation,
    reset_counters as reset_validation_counters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def split_categories(values: Sequence[str]) -> list[str]:
    categories: list[str] = []
    for value in values:
        categories.extend(item.strip() for item in value.split(",") if item.strip())
    return categories


def count_task_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.iterdir() if child.is_dir() and child.name.startswith("task_"))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "seed": run_dir / "seed.json",
        "candidates": run_dir / "candidates",
        "accepted": run_dir / "accepted",
        "logs": run_dir / "pipeline_logs",
        "state": run_dir / "pipeline_state",
    }


def append_prepare_filters(args: argparse.Namespace, command: list[str]) -> None:
    command.extend(["--start", str(args.start)])
    if args.min_score is not None:
        command.extend(["--min-score", str(args.min_score)])
    for category in split_categories(args.category or []):
        command.extend(["--category", category])
    if args.per_category is not None:
        command.extend(["--per-category", str(args.per_category)])
    elif args.sample_size is not None:
        command.extend(["--sample-size", str(args.sample_size), "--distribution", args.distribution])
    elif args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.sort_by_score:
        command.append("--sort-by-score")


def run_command(
    name: str,
    command: list[str],
    log_path: Path,
    marker_path: Path,
    resume: bool,
    cwd: Path = REPO_ROOT,
) -> dict[str, Any]:
    if resume and marker_path.exists():
        marker = load_json(marker_path)
        if marker.get("status") == "completed":
            return {"stage": name, "status": "skipped", "marker": str(marker_path)}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()

    marker = {
        "stage": name,
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "command": command,
        "log": str(log_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(marker_path, marker)
    if returncode != 0:
        raise SystemExit(f"Stage {name!r} failed with return code {returncode}. See {log_path}")
    return marker


def build_summary(args: argparse.Namespace, paths: dict[str, Path], stages: list[dict[str, Any]]) -> dict[str, Any]:
    validation_report = load_json(paths["accepted"] / "validation_report.json")
    return {
        "mode": "streaming",
        "run_dir": str(args.output),
        "input": str(args.input),
        "seed": str(paths["seed"]),
        "candidates": str(paths["candidates"]),
        "accepted": str(paths["accepted"]),
        "counts": {
            "candidates": count_task_dirs(paths["candidates"]),
            "accepted": count_task_dirs(paths["accepted"]),
        },
        "reports": {
            "generation": str(paths["candidates"] / "generation_summary.json"),
            "validation": str(paths["accepted"] / "validation_report.json"),
            "validate_log": str(paths["accepted"] / "validate_tasks.log"),
        },
        "validation_status_counts": validation_report.get("status_counts", {}),
        "stages": stages,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": "validate_each_candidate_immediately_accept_reward_1_skip_failed",
    }


def count_seed_questions(seed_path: Path) -> int:
    data = load_json(seed_path)
    return sum(1 for row in data.get("questions", []) if row.get("accepted_answer"))


def run_streaming_generation_and_validation(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Overlap unchanged task generation with unchanged Docker validation."""
    marker_path = paths["state"] / "02_generate_and_validate.json"
    if args.resume and marker_path.exists():
        marker = load_json(marker_path)
        if marker.get("status") == "completed":
            return {
                "stage": "generate_and_validate",
                "status": "skipped",
                "marker": str(marker_path),
            }

    paths["accepted"].mkdir(parents=True, exist_ok=True)
    add_output_file_logger(paths["logs"] / "02_generate_and_validate.log")
    add_output_file_logger(paths["accepted"] / "validate_tasks.log")
    reset_validation_counters()

    instruction_rewrite_mode = os.environ.get(
        "INSTRUCTION_REWRITE_MODE",
        task_generator.DEFAULT_INSTRUCTION_REWRITE_MODE,
    )
    lossy_instruction_ratio = float(
        os.environ.get(
            "LOSSY_INSTRUCTION_RATIO",
            task_generator.DEFAULT_LOSSY_INSTRUCTION_RATIO,
        )
    )
    task_generator.configure_generation(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        instruction_rewrite_mode=instruction_rewrite_mode,
        lossy_instruction_ratio=lossy_instruction_ratio,
    )

    expected_total = count_seed_questions(paths["seed"])
    validator = TaskValidator(paths["accepted"], args.timeout)
    validation_futures: dict[Future[dict], Path] = {}
    validation_started_at: float | None = None
    started_at = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.validate_workers) as validation_executor:

            def schedule_validation(task_dir: Path, _generation_total: int) -> None:
                nonlocal validation_started_at
                if validation_started_at is None:
                    validation_started_at = time.time()
                future = validation_executor.submit(
                    validator.validate,
                    task_dir,
                    expected_total,
                )
                validation_futures[future] = task_dir

            if args.resume and paths["candidates"].exists():
                for task_dir in sorted(paths["candidates"].glob("task_*")):
                    if task_dir.is_dir() and is_complete_task_dir(task_dir):
                        schedule_validation(task_dir, expected_total)

            generation_summary = task_generator.run_generation(
                input_path=paths["seed"],
                output_dir=paths["candidates"],
                workers=args.generate_workers,
                resume=args.resume,
                on_generated=schedule_validation,
            )
            generation_finished_at = time.time()

            validation_results: list[dict[str, Any]] = []
            for future in as_completed(validation_futures):
                task_dir = validation_futures[future]
                try:
                    validation_results.append(future.result())
                except Exception as exc:
                    validation_results.append(
                        {
                            "task": task_dir.name,
                            "status": "error",
                            "error": str(exc),
                        }
                    )

        finished_at = time.time()
        validation_results.sort(key=lambda row: str(row.get("task", "")))
        validation_elapsed = (
            finished_at - validation_started_at
            if validation_started_at is not None
            else 0.0
        )
        validation_report = finalize_validation(
            paths["accepted"],
            validation_results,
            validation_elapsed,
        )
        marker = {
            "stage": "generate_and_validate",
            "status": "completed",
            "elapsed_seconds": round(finished_at - started_at, 3),
            "generation_elapsed_seconds": round(
                generation_finished_at - started_at,
                3,
            ),
            "validation_elapsed_seconds": round(validation_elapsed, 3),
            "generate_workers": args.generate_workers,
            "validate_workers": args.validate_workers,
            "generation": generation_summary,
            "validation_status_counts": validation_report["status_counts"],
            "log": str(paths["logs"] / "02_generate_and_validate.log"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(marker_path, marker)
        return marker
    except Exception as exc:
        marker = {
            "stage": "generate_and_validate",
            "status": "failed",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "error": str(exc),
            "log": str(paths["logs"] / "02_generate_and_validate.log"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(marker_path, marker)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input StackOverflow JSONL dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Pipeline run directory.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="Skip completed prepare/generate/validate stages.")

    parser.add_argument("--generate-workers", type=int, default=1)
    parser.add_argument("--validate-workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)

    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))

    parser.add_argument("--min-score", type=int, default=None)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--distribution", default="terminal-bench-2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sort-by-score", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.output.resolve()
    paths = build_paths(run_dir)
    for path in (run_dir, paths["logs"], paths["state"]):
        path.mkdir(parents=True, exist_ok=True)

    py = args.python_bin
    stages: list[dict[str, Any]] = []

    prepare = [
        py,
        "-m",
        "sources.stackoverflow.prepare_dataset",
        "--input",
        str(args.input.resolve()),
        "--output",
        str(paths["seed"]),
        "--format",
        "generator-json",
    ]
    append_prepare_filters(args, prepare)
    stages.append(run_command("prepare_seed", prepare, paths["logs"] / "01_prepare_seed.log", paths["state"] / "01_prepare_seed.json", args.resume))

    stages.append(run_streaming_generation_and_validation(args, paths))

    summary = build_summary(args, paths, stages)
    write_json(run_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
