from __future__ import annotations

import json
from pathlib import Path

from generator.failure.classifier import classify_failure


def write_log(log_dir: Path, name: str, text: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / name).write_text(text, encoding="utf-8")


def test_classify_runtime_missing_from_build_failure(tmp_path: Path) -> None:
    log_dir = tmp_path / "validation_logs" / "task_00001"
    write_log(log_dir, "result.json", json.dumps({"status": "build_failed", "error": "Unable to locate package dotnet"}))
    write_log(log_dir, "build.stderr", "Unable to locate package dotnet-sdk")

    diagnosis = classify_failure("task_00001", log_dir)

    assert diagnosis.failure_class == "runtime_missing"
    assert diagnosis.allowed_repairs == ["Dockerfile"]


def test_classify_solution_path_mismatch(tmp_path: Path) -> None:
    log_dir = tmp_path / "validation_logs" / "task_00002"
    write_log(log_dir, "result.json", json.dumps({"status": "failed"}))
    write_log(log_dir, "solve.stdout", "cd: /app/task_file/input/repo: No such file or directory")

    diagnosis = classify_failure("task_00002", log_dir)

    assert diagnosis.failure_class == "path_mismatch"
    assert "solution/solve.sh" in diagnosis.allowed_repairs


def test_classify_missing_verifier_python_as_runtime_missing(tmp_path: Path) -> None:
    log_dir = tmp_path / "validation_logs" / "task_00003"
    write_log(log_dir, "result.json", json.dumps({"status": "failed"}))
    write_log(log_dir, "test.stdout", "Error: no Python 3.12+ with pytest is installed.")

    diagnosis = classify_failure("task_00003", log_dir)

    assert diagnosis.failure_class == "runtime_missing"
    assert diagnosis.allowed_repairs == ["environment/Dockerfile"]
