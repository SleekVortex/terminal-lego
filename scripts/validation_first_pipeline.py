#!/usr/bin/env python3
"""Helpers for the validation-first Terminal-Lego task filtering flow."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PASSED_STATUS = "passed"
REQUIRED_FOR_REGEN = (
    "instruction.md",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test_outputs.py",
    "tests/test.sh",
)


def load_validation_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def has_required_regen_artifacts(task_dir: Path) -> bool:
    return all((task_dir / rel).exists() for rel in REQUIRED_FOR_REGEN)


def collect_failures(args: argparse.Namespace) -> None:
    report = load_validation_report(args.validation_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in report.get("results", []):
        task_name = result.get("task")
        if not task_name or result.get("status") == PASSED_STATUS:
            continue
        task_dir = args.tasks_dir / task_name
        regen_eligible = has_required_regen_artifacts(task_dir)
        row = {
            "task": task_name,
            "baseline_status": result.get("status"),
            "baseline_reward": result.get("reward"),
            "baseline_error": result.get("error"),
            "regen_eligible": regen_eligible,
            "final_status": "needs_dockerfile_regen" if regen_eligible else "discard_missing_artifacts",
        }
        rows.append(row)

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "validation_report": str(args.validation_report),
        "tasks_dir": str(args.tasks_dir),
        "failed_total": len(rows),
        "regen_eligible": sum(1 for row in rows if row["regen_eligible"]),
        "discard_missing_artifacts": sum(1 for row in rows if not row["regen_eligible"]),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def copy_task(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def parse_extra_validated_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected SOURCE=DIR, got: {value!r}")
    source, raw_path = value.split("=", 1)
    source = source.strip()
    if not source:
        raise ValueError(f"Empty source name in SOURCE=DIR item: {value!r}")
    return source, Path(raw_path)


def merge_accepted(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    copied = 0
    duplicates = 0
    seen = set()

    sources = [
        ("baseline", args.baseline_validated_dir),
        ("regenerated", args.regenerated_validated_dir),
        ("repair", getattr(args, "repair_validated_dir", None)),
    ]
    for item in getattr(args, "extra_validated_dir", None) or []:
        sources.append(parse_extra_validated_dir(item))

    for source_name, source_dir in sources:
        if source_dir is None or not source_dir.exists():
            continue
        for task_dir in sorted(path for path in source_dir.iterdir() if path.is_dir() and path.name.startswith("task_")):
            duplicate = task_dir.name in seen
            if duplicate:
                duplicates += 1
            seen.add(task_dir.name)
            copied_ok = copy_task(task_dir, args.output_dir / task_dir.name)
            copied += int(copied_ok and not duplicate)
            rows.append({
                "task": task_dir.name,
                "source": source_name,
                "copied": copied_ok,
                "duplicate": duplicate,
            })

    report = {
        "baseline_validated_dir": str(args.baseline_validated_dir),
        "regenerated_validated_dir": str(args.regenerated_validated_dir) if args.regenerated_validated_dir else None,
        "repair_validated_dir": str(getattr(args, "repair_validated_dir", None)) if getattr(args, "repair_validated_dir", None) else None,
        "extra_validated_dir": getattr(args, "extra_validated_dir", None) or [],
        "output_dir": str(args.output_dir),
        "final_accepted": copied,
        "duplicates": duplicates,
        "rows": rows,
    }
    report_path = args.output_dir / "final_accepted_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "final_accepted": copied,
        "duplicates": duplicates,
        "report": str(report_path),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect-failures", help="Create JSONL failed-task list from validation_report.json.")
    collect.add_argument("--validation-report", required=True, type=Path)
    collect.add_argument("--tasks-dir", required=True, type=Path)
    collect.add_argument("--output", required=True, type=Path)
    collect.set_defaults(func=collect_failures)

    merge = subparsers.add_parser("merge-accepted", help="Merge baseline and regenerated validated tasks.")
    merge.add_argument("--baseline-validated-dir", required=True, type=Path)
    merge.add_argument("--regenerated-validated-dir", type=Path, default=None)
    merge.add_argument("--repair-validated-dir", type=Path, default=None)
    merge.add_argument("--extra-validated-dir", action="append", default=[], help="Additional accepted source as SOURCE=DIR.")
    merge.add_argument("--output-dir", required=True, type=Path)
    merge.set_defaults(func=merge_accepted)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
