from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from generator.contracts import FailureDiagnosis


def read_tail(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def load_result(log_dir: Path) -> dict[str, Any]:
    path = log_dir / "result.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def classify_failure(task_name: str, log_dir: Path) -> FailureDiagnosis:
    result = load_result(log_dir)
    status = result.get("status") or "unknown"
    build_text = read_tail(log_dir / "build.stderr") + "\n" + read_tail(log_dir / "build.stdout")
    solve_text = read_tail(log_dir / "solve.stderr") + "\n" + read_tail(log_dir / "solve.stdout")
    test_text = read_tail(log_dir / "test.stderr") + "\n" + read_tail(log_dir / "test.stdout")
    combined = "\n".join([str(result.get("error") or ""), build_text, solve_text, test_text])

    if status == "build_failed":
        if re.search(r"not found|Unable to locate package|command not found|No such file", combined, re.I):
            return FailureDiagnosis(task_name, "runtime_missing", 0.8, evidence=[combined[-800:]], allowed_repairs=["Dockerfile"])
        return FailureDiagnosis(task_name, "build_failed", 0.75, evidence=[combined[-800:]], allowed_repairs=["Dockerfile"])

    if "No such file or directory" in combined or "does not exist" in combined:
        return FailureDiagnosis(
            task_name,
            "path_mismatch",
            0.8,
            evidence=[combined[-800:]],
            allowed_repairs=["solution/solve.sh", "environment"],
        )

    if re.search(r"command not found|/usr/share/dotnet|Cannot find module|ModuleNotFoundError", combined, re.I):
        return FailureDiagnosis(
            task_name,
            "runtime_missing",
            0.8,
            evidence=[combined[-800:]],
            allowed_repairs=["environment/Dockerfile"],
        )

    solve_result = log_dir / "solve.json"
    if solve_result.exists():
        try:
            solve_data = json.loads(solve_result.read_text(encoding="utf-8"))
            if solve_data.get("returncode") not in (0, "0"):
                return FailureDiagnosis(
                    task_name,
                    "solution_error",
                    0.75,
                    evidence=[solve_text[-800:]],
                    allowed_repairs=["solution/solve.sh"],
                )
        except json.JSONDecodeError:
            pass

    if "FAILED" in test_text or "AssertionError" in test_text:
        return FailureDiagnosis(
            task_name,
            "test_assertion_or_error",
            0.65,
            evidence=[test_text[-800:]],
            allowed_repairs=["solution/solve.sh"],
        )

    if "timeout" in combined.lower():
        return FailureDiagnosis(task_name, "timeout", 0.6, evidence=[combined[-800:]], allowed_repairs=["solution/solve.sh", "environment/Dockerfile"])

    return FailureDiagnosis(task_name, "unknown", 0.3, evidence=[combined[-800:]], allowed_repairs=[])


def classify_from_validation_report(validation_output_dir: Path) -> list[FailureDiagnosis]:
    report_path = validation_output_dir / "validation_report.json"
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnoses: list[FailureDiagnosis] = []
    for row in report.get("results", []):
        if row.get("status") == "passed":
            continue
        task = row.get("task")
        log_dir = row.get("log_dir")
        if not task or not log_dir:
            continue
        diagnoses.append(classify_failure(str(task), validation_output_dir / str(log_dir)))
    return diagnoses
