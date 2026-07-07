#!/usr/bin/env python3
"""Run the full Terminal-Lego task generation pipeline.

Pipeline:
1. prepare StackOverflow seed
2. generate task candidates
3. validate baseline candidates
4. collect failed tasks
5. regenerate Dockerfiles for eligible failures
6. validate regenerated tasks
7. diagnose remaining failures
8. repair diagnosed failures with validation
9. merge all accepted tasks
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
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if resume and marker_path.exists():
        marker = load_json(marker_path)
        if marker.get("status") == "completed":
            return {"stage": name, "status": "skipped", "marker": str(marker_path)}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged_env,
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

    elapsed = round(time.time() - started, 3)
    marker = {
        "stage": name,
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "command": command,
        "log": str(log_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(marker_path, marker)
    if returncode != 0:
        raise SystemExit(f"Stage {name!r} failed with return code {returncode}. See {log_path}")
    return marker


def filter_jsonl(input_path: Path, output_path: Path, key: str, expected_value: Any) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        if input_path.exists():
            for line in input_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get(key) == expected_value:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    return count


def build_summary(args: argparse.Namespace, paths: dict[str, Path], stages: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_report = load_json(paths["baseline_validation"] / "validation_report.json")
    regenerated_report = load_json(paths["regenerated_validation"] / "validation_report.json")
    repair_summary = load_json(paths["repair"] / "repair_summary.json")
    accepted_report = load_json(paths["accepted"] / "final_accepted_report.json")
    return {
        "run_dir": str(args.output),
        "input": str(args.input),
        "seed": str(paths["seed"]),
        "candidates": str(paths["candidates"]),
        "accepted": str(paths["accepted"]),
        "counts": {
            "candidates": count_task_dirs(paths["candidates"]),
            "baseline_accepted": count_task_dirs(paths["baseline_validation"]),
            "regenerated_accepted": count_task_dirs(paths["regenerated_validation"]),
            "repair_accepted": count_task_dirs(paths["repair"] / "repaired_validated"),
            "final_accepted": accepted_report.get("final_accepted", count_task_dirs(paths["accepted"])),
        },
        "reports": {
            "baseline_validation": str(paths["baseline_validation"] / "validation_report.json"),
            "baseline_failures": str(paths["baseline_failures"]),
            "dockerfile_regen": str(paths["dockerfile_regen"] / "regenerate_dockerfiles_summary.json"),
            "regenerated_validation": str(paths["regenerated_validation"] / "validation_report.json"),
            "regenerated_failures": str(paths["regenerated_failures"]),
            "diagnoses": str(paths["diagnoses"]),
            "repair": str(paths["repair"] / "repair_summary.json"),
            "accepted": str(paths["accepted"] / "final_accepted_report.json"),
        },
        "baseline_status_counts": baseline_report.get("status_counts", {}),
        "regenerated_status_counts": regenerated_report.get("status_counts", {}),
        "repair_status_counts": repair_summary.get("status_counts", {}),
        "stages": stages,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input StackOverflow JSONL dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Pipeline run directory.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="Skip stages with completed markers.")

    parser.add_argument("--generate-workers", type=int, default=1)
    parser.add_argument("--validate-workers", type=int, default=8)
    parser.add_argument("--dockerfile-workers", type=int, default=8)
    parser.add_argument("--repair-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--repair-max-attempts", type=int, default=2)

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

    parser.add_argument("--skip-dockerfile-regeneration", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.output.resolve()
    paths = {
        "seed": run_dir / "seed.json",
        "candidates": run_dir / "candidates",
        "baseline_validation": run_dir / "validation" / "baseline",
        "baseline_failures": run_dir / "failures" / "baseline_failures.jsonl",
        "dockerfile_regen_tasks": run_dir / "failures" / "dockerfile_regen_tasks.jsonl",
        "dockerfile_regen": run_dir / "dockerfile_regen",
        "regenerated_validation": run_dir / "validation" / "dockerfile_regen",
        "regenerated_failures": run_dir / "failures" / "regenerated_failures.jsonl",
        "diagnoses": run_dir / "diagnoses" / "remaining_failures.jsonl",
        "repair": run_dir / "repair",
        "accepted": run_dir / "accepted",
        "logs": run_dir / "pipeline_logs",
        "state": run_dir / "pipeline_state",
    }
    for path in (run_dir, paths["logs"], paths["state"]):
        path.mkdir(parents=True, exist_ok=True)

    py = args.python_bin
    stages: list[dict[str, Any]] = []

    prepare = [py, "-m", "sources.stackoverflow.prepare_dataset", "--input", str(args.input.resolve()), "--output", str(paths["seed"]), "--format", "generator-json"]
    append_prepare_filters(args, prepare)
    stages.append(run_command("prepare_seed", prepare, paths["logs"] / "01_prepare_seed.log", paths["state"] / "01_prepare_seed.json", args.resume))

    generate = [py, "-m", "generator.task_generator", "--input", str(paths["seed"]), "--output", str(paths["candidates"]), "--workers", str(args.generate_workers), "--api-key", args.api_key]
    if args.api_base:
        generate.extend(["--api-base", args.api_base])
    if args.model:
        generate.extend(["--model", args.model])
    if args.resume:
        generate.append("--resume")
    stages.append(run_command("generate_candidates", generate, paths["logs"] / "02_generate_candidates.log", paths["state"] / "02_generate_candidates.json", args.resume))

    baseline_validate = [py, "-m", "validator.validate_tasks", "--input", str(paths["candidates"]), "--output", str(paths["baseline_validation"]), "--workers", str(args.validate_workers), "--timeout", str(args.timeout)]
    stages.append(run_command("validate_baseline", baseline_validate, paths["logs"] / "03_validate_baseline.log", paths["state"] / "03_validate_baseline.json", args.resume))

    collect_baseline = [py, "-m", "scripts.validation_first_pipeline", "collect-failures", "--validation-report", str(paths["baseline_validation"] / "validation_report.json"), "--tasks-dir", str(paths["candidates"]), "--output", str(paths["baseline_failures"])]
    stages.append(run_command("collect_baseline_failures", collect_baseline, paths["logs"] / "04_collect_baseline_failures.log", paths["state"] / "04_collect_baseline_failures.json", args.resume))

    regen_task_count = filter_jsonl(paths["baseline_failures"], paths["dockerfile_regen_tasks"], "regen_eligible", True)
    write_json(paths["state"] / "05_dockerfile_regen_tasks.json", {"stage": "dockerfile_regen_tasks", "count": regen_task_count, "path": str(paths["dockerfile_regen_tasks"])})

    if not args.skip_dockerfile_regeneration and regen_task_count > 0:
        regen = [py, "-m", "scripts.regenerate_dockerfiles", "--tasks-dir", str(paths["candidates"]), "--report-dir", str(paths["dockerfile_regen"]), "--workers", str(args.dockerfile_workers), "--task-list", str(paths["dockerfile_regen_tasks"]), "--api-key", args.api_key]
        if args.api_base:
            regen.extend(["--api-base", args.api_base])
        if args.model:
            regen.extend(["--model", args.model])
        stages.append(run_command("regenerate_dockerfiles", regen, paths["logs"] / "05_regenerate_dockerfiles.log", paths["state"] / "05_regenerate_dockerfiles.json", args.resume))

        validate_regenerated = [py, "-m", "validator.validate_tasks", "--input", str(paths["candidates"]), "--output", str(paths["regenerated_validation"]), "--workers", str(args.validate_workers), "--timeout", str(args.timeout), "--task-list", str(paths["dockerfile_regen_tasks"])]
        stages.append(run_command("validate_regenerated", validate_regenerated, paths["logs"] / "06_validate_regenerated.log", paths["state"] / "06_validate_regenerated.json", args.resume))

        collect_regenerated = [py, "-m", "scripts.validation_first_pipeline", "collect-failures", "--validation-report", str(paths["regenerated_validation"] / "validation_report.json"), "--tasks-dir", str(paths["candidates"]), "--output", str(paths["regenerated_failures"])]
        stages.append(run_command("collect_regenerated_failures", collect_regenerated, paths["logs"] / "07_collect_regenerated_failures.log", paths["state"] / "07_collect_regenerated_failures.json", args.resume))

        diagnose = [py, "-m", "scripts.diagnose_failed_tasks", "--validation-output", str(paths["regenerated_validation"]), "--output", str(paths["diagnoses"])]
        stages.append(run_command("diagnose_remaining_failures", diagnose, paths["logs"] / "08_diagnose_remaining_failures.log", paths["state"] / "08_diagnose_remaining_failures.json", args.resume))
    else:
        paths["regenerated_validation"].mkdir(parents=True, exist_ok=True)
        stages.append({"stage": "regenerate_dockerfiles", "status": "skipped", "reason": "disabled_or_no_eligible_failures"})
        if args.skip_dockerfile_regeneration:
            diagnose = [py, "-m", "scripts.diagnose_failed_tasks", "--validation-output", str(paths["baseline_validation"]), "--output", str(paths["diagnoses"])]
            stages.append(run_command("diagnose_baseline_failures", diagnose, paths["logs"] / "08_diagnose_baseline_failures.log", paths["state"] / "08_diagnose_baseline_failures.json", args.resume))
        else:
            paths["diagnoses"].parent.mkdir(parents=True, exist_ok=True)
            paths["diagnoses"].write_text("", encoding="utf-8")

    diagnosis_count = sum(1 for line in paths["diagnoses"].read_text(encoding="utf-8").splitlines() if line.strip()) if paths["diagnoses"].exists() else 0
    if not args.skip_repair and diagnosis_count > 0:
        repair = [py, "-m", "scripts.repair_failed_tasks", "--tasks-dir", str(paths["candidates"]), "--output", str(paths["repair"]), "--diagnoses", str(paths["diagnoses"]), "--workers", str(args.repair_workers), "--max-attempts", str(args.repair_max_attempts), "--timeout", str(args.timeout), "--api-key", args.api_key]
        if args.api_base:
            repair.extend(["--api-base", args.api_base])
        if args.model:
            repair.extend(["--model", args.model])
        stages.append(run_command("repair_failures", repair, paths["logs"] / "09_repair_failures.log", paths["state"] / "09_repair_failures.json", args.resume))
    else:
        (paths["repair"] / "repaired_validated").mkdir(parents=True, exist_ok=True)
        stages.append({"stage": "repair_failures", "status": "skipped", "reason": "disabled_or_no_diagnoses"})

    merge = [
        py,
        "-m",
        "scripts.validation_first_pipeline",
        "merge-accepted",
        "--baseline-validated-dir",
        str(paths["baseline_validation"]),
        "--regenerated-validated-dir",
        str(paths["regenerated_validation"]),
        "--repair-validated-dir",
        str(paths["repair"] / "repaired_validated"),
        "--output-dir",
        str(paths["accepted"]),
    ]
    stages.append(run_command("merge_accepted", merge, paths["logs"] / "10_merge_accepted.log", paths["state"] / "10_merge_accepted.json", False))

    summary = build_summary(args, paths, stages)
    write_json(run_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
