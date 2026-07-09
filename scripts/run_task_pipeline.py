#!/usr/bin/env python3
"""Run the simple production Terminal-Lego task pipeline.

Production policy:
1. prepare StackOverflow seed
2. generate task candidates once
3. validate candidates with Docker
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
from pathlib import Path
from typing import Any, Sequence


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
        "mode": "simple",
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
            "validation": str(paths["accepted"] / "validation_report.json"),
            "validate_log": str(paths["accepted"] / "validate_tasks.log"),
        },
        "validation_status_counts": validation_report.get("status_counts", {}),
        "stages": stages,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": "generate_once_validate_accept_reward_1_skip_failed",
    }


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

    # Compatibility no-ops for commands written against the experimental pipeline.
    parser.add_argument("--mode", choices=("simple", "staged", "streaming"), default="simple")
    parser.add_argument("--dockerfile-workers", type=int, default=0)
    parser.add_argument("--repair-workers", type=int, default=0)
    parser.add_argument("--task-workers", type=int, default=None)
    parser.add_argument("--llm-concurrency", type=int, default=None)
    parser.add_argument("--docker-concurrency", type=int, default=None)
    parser.add_argument("--existing-candidates-only", action="store_true")
    parser.add_argument("--repair-max-attempts", type=int, default=0)
    parser.add_argument("--skip-dockerfile-regeneration", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.output.resolve()
    paths = build_paths(run_dir)
    for path in (run_dir, paths["logs"], paths["state"]):
        path.mkdir(parents=True, exist_ok=True)

    if args.existing_candidates_only:
        raise SystemExit("--existing-candidates-only is not supported by the simple pipeline")

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

    generate = [
        py,
        "-m",
        "generator.task_generator",
        "--input",
        str(paths["seed"]),
        "--output",
        str(paths["candidates"]),
        "--workers",
        str(args.generate_workers),
        "--api-key",
        args.api_key,
    ]
    if args.api_base:
        generate.extend(["--api-base", args.api_base])
    if args.model:
        generate.extend(["--model", args.model])
    if args.resume:
        generate.append("--resume")
    stages.append(run_command("generate_candidates", generate, paths["logs"] / "02_generate_candidates.log", paths["state"] / "02_generate_candidates.json", args.resume))

    validate = [
        py,
        "-m",
        "validator.validate_tasks",
        "--input",
        str(paths["candidates"]),
        "--output",
        str(paths["accepted"]),
        "--workers",
        str(args.validate_workers),
        "--timeout",
        str(args.timeout),
    ]
    stages.append(run_command("validate_candidates", validate, paths["logs"] / "03_validate_candidates.log", paths["state"] / "03_validate_candidates.json", args.resume))

    summary = build_summary(args, paths, stages)
    write_json(run_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
