from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from generator.agents.base import LLMCall
from generator.contracts import FailureDiagnosis, RepairAttempt
from generator.failure.classifier import classify_failure
from generator.llm_client import call_llm_api
from generator.repair.agent import RepairAgent


ValidateTask = Callable[[Path, Path, int], dict[str, Any]]


def load_diagnoses(path: Path) -> list[FailureDiagnosis]:
    rows: list[FailureDiagnosis] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        rows.append(
            FailureDiagnosis(
                task_name=str(data.get("task_name") or data.get("task")),
                failure_class=str(data.get("failure_class") or data.get("final_status") or "unknown"),
                confidence=float(data.get("confidence") or 0.0),
                evidence=list(data.get("evidence") or [str(data.get("baseline_error") or "")]),
                allowed_repairs=list(data.get("allowed_repairs") or []),
            )
        )
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class RepairLoop:
    def __init__(
        self,
        tasks_dir: Path,
        output_dir: Path,
        max_attempts: int = 2,
        validate: bool = True,
        timeout: int = 300,
        llm_call: LLMCall = call_llm_api,
        validate_task: ValidateTask | None = None,
    ):
        self.tasks_dir = tasks_dir
        self.output_dir = output_dir
        self.max_attempts = max_attempts
        self.validate = validate
        self.timeout = timeout
        self.llm_call = llm_call
        self.validate_task = validate_task
        self.attempts_dir = output_dir / "attempts"
        self.validated_dir = output_dir / "repaired_validated"

    def run_one(self, diagnosis: FailureDiagnosis) -> list[RepairAttempt]:
        attempts: list[RepairAttempt] = []
        source_task_dir = self.tasks_dir / diagnosis.task_name
        if not source_task_dir.exists():
            return [
                RepairAttempt(
                    task_name=diagnosis.task_name,
                    attempt=0,
                    failure_class=diagnosis.failure_class,
                    status="missing_task",
                    changed_files=[],
                    error=f"task dir does not exist: {source_task_dir}",
                )
            ]

        current_source = source_task_dir
        current_diagnosis = diagnosis
        for attempt_idx in range(1, self.max_attempts + 1):
            attempt_root = self.attempts_dir / diagnosis.task_name / f"attempt_{attempt_idx:02d}"
            repaired_task_dir = attempt_root / diagnosis.task_name
            agent = RepairAgent(self.llm_call, diagnosis.task_name)
            changed_files, note = agent.repair(current_source, repaired_task_dir, current_diagnosis)
            status = "patched" if changed_files else "not_repaired"
            validation_result: dict[str, Any] = {}
            if changed_files and self.validate:
                if self.validate_task is None:
                    raise RuntimeError("validate=True requires a validate_task callback")
                validation_output = attempt_root / "validation"
                validation_result = self.validate_task(repaired_task_dir, validation_output, self.timeout)
                if validation_result.get("status") == "passed" and validation_result.get("reward") == 1.0:
                    status = "accepted"
                    accepted_dst = self.validated_dir / diagnosis.task_name
                    if accepted_dst.exists():
                        shutil.rmtree(accepted_dst)
                    shutil.copytree(repaired_task_dir, accepted_dst)
                    break_attempt = RepairAttempt(
                        task_name=diagnosis.task_name,
                        attempt=attempt_idx,
                        failure_class=current_diagnosis.failure_class,
                        status=status,
                        changed_files=changed_files,
                        validation_result=validation_result,
                        error=None,
                    )
                    attempts.append(break_attempt)
                    return attempts
                current_source = repaired_task_dir
                log_dir = validation_result.get("log_dir")
                if log_dir:
                    current_diagnosis = classify_failure(diagnosis.task_name, validation_output / str(log_dir))
            attempts.append(
                RepairAttempt(
                    task_name=diagnosis.task_name,
                    attempt=attempt_idx,
                    failure_class=current_diagnosis.failure_class,
                    status=status,
                    changed_files=changed_files,
                    validation_result=validation_result,
                    error=None if changed_files else note,
                )
            )
            if not changed_files:
                return attempts
        return attempts


def run_repair_loop(
    tasks_dir: Path,
    diagnoses: Iterable[FailureDiagnosis],
    output_dir: Path,
    max_attempts: int = 2,
    workers: int = 1,
    validate: bool = True,
    timeout: int = 300,
    llm_call: LLMCall = call_llm_api,
    validate_task: ValidateTask | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "repair_report.jsonl"
    if report_path.exists():
        report_path.unlink()

    diagnoses = list(diagnoses)
    loop = RepairLoop(
        tasks_dir,
        output_dir,
        max_attempts=max_attempts,
        validate=validate,
        timeout=timeout,
        llm_call=llm_call,
        validate_task=validate_task,
    )
    attempts: list[RepairAttempt] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(loop.run_one, diagnosis): diagnosis for diagnosis in diagnoses}
        for future in as_completed(futures):
            for attempt in future.result():
                attempts.append(attempt)
                append_jsonl(report_path, attempt.to_dict())

    status_counts: dict[str, int] = {}
    for attempt in attempts:
        status_counts[attempt.status] = status_counts.get(attempt.status, 0) + 1
    summary = {
        "tasks_dir": str(tasks_dir),
        "output_dir": str(output_dir),
        "total_diagnoses": len(diagnoses),
        "total_attempts": len(attempts),
        "status_counts": status_counts,
        "report": str(report_path),
    }
    (output_dir / "repair_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
