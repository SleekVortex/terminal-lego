#!/usr/bin/env python3
"""Repair failed Terminal-Lego task candidates from failure diagnoses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from generator import llm_client
from generator.failure.classifier import classify_from_validation_report
from generator.llm_client import DEFAULT_API_BASE, DEFAULT_MODEL, TokenTracker
from generator.repair.loop import load_diagnoses, run_repair_loop
from validator.validate_tasks import TaskValidator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnoses", type=Path, default=None, help="JSONL from diagnose_failed_tasks.py")
    parser.add_argument("--validation-output", type=Path, default=None, help="Validation output dir with validation_report.json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-validate", action="store_true", help="Only materialize repaired attempts; do not run Docker validation.")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    args = parser.parse_args()

    if not args.diagnoses and not args.validation_output:
        parser.error("Provide --diagnoses or --validation-output")

    llm_client.token_tracker = TokenTracker(str(args.output / "repair_token_usage.json"))
    llm_client._config["api_base"] = args.api_base
    llm_client._config["api_key"] = args.api_key
    llm_client._config["model"] = args.model

    if args.diagnoses:
        diagnoses = load_diagnoses(args.diagnoses)
    else:
        diagnoses = classify_from_validation_report(args.validation_output)

    def validate_task(task_dir: Path, validation_output: Path, timeout: int) -> dict:
        return TaskValidator(validation_output, timeout).validate(task_dir, total=1)

    summary = run_repair_loop(
        tasks_dir=args.tasks_dir,
        diagnoses=diagnoses,
        output_dir=args.output,
        max_attempts=args.max_attempts,
        workers=args.workers,
        validate=not args.no_validate,
        timeout=args.timeout,
        llm_call=llm_client.call_llm_api,
        validate_task=None if args.no_validate else validate_task,
    )
    summary["model"] = args.model
    summary["api_base"] = args.api_base
    summary["token_usage"] = llm_client.token_tracker.get_summary()
    (args.output / "repair_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
