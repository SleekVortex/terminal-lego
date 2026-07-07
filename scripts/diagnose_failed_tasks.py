#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from generator.failure.classifier import classify_from_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify failed validated tasks from per-task validation logs.")
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    diagnoses = classify_from_validation_report(args.validation_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for diagnosis in diagnoses:
            handle.write(json.dumps(diagnosis.to_dict(), ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for diagnosis in diagnoses:
        counts[diagnosis.failure_class] = counts.get(diagnosis.failure_class, 0) + 1
    summary = {
        "validation_output": str(args.validation_output),
        "output": str(args.output),
        "total": len(diagnoses),
        "failure_class_counts": counts,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
