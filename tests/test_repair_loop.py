from __future__ import annotations

import json
from pathlib import Path

from generator.contracts import FailureDiagnosis
from generator.repair.agent import is_allowed_file, normalize_allowed_repairs
from generator.repair.loop import run_repair_loop


def write_task(task_dir: Path) -> None:
    (task_dir / "environment" / "task_file").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("# Task\n", encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim-bookworm\nWORKDIR /app\nCOPY ./task_file /app/task_file\n",
        encoding="utf-8",
    )
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    (task_dir / "tests" / "test_outputs.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\npython -m pytest /tests/test_outputs.py\n", encoding="utf-8")


def test_normalize_allowed_repairs() -> None:
    allowed = normalize_allowed_repairs(["Dockerfile", "solution", "environment", "../bad"])

    assert "environment/Dockerfile" in allowed
    assert "solution/solve.sh" in allowed
    assert "environment/**" in allowed
    assert not is_allowed_file("../bad", allowed)
    assert is_allowed_file("environment/task_file/input.txt", allowed)


def test_repair_loop_patches_allowed_file_without_validation(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task_00001"
    write_task(task_dir)
    diagnosis = FailureDiagnosis(
        task_name="task_00001",
        failure_class="solution_error",
        confidence=0.9,
        evidence=["solve.sh exited 1"],
        allowed_repairs=["solution/solve.sh"],
    )

    def fake_llm(*args, **kwargs):
        return "```json\n" + json.dumps({"files": {"solution/solve.sh": "#!/bin/bash\ntrue\n"}, "notes": "fixed"}) + "\n```"

    summary = run_repair_loop(
        tasks_dir=tasks_dir,
        diagnoses=[diagnosis],
        output_dir=tmp_path / "repair",
        max_attempts=1,
        validate=False,
        llm_call=fake_llm,
    )

    repaired = tmp_path / "repair" / "attempts" / "task_00001" / "attempt_01" / "task_00001" / "solution" / "solve.sh"
    assert repaired.read_text(encoding="utf-8") == "#!/bin/bash\ntrue\n"
    assert summary["status_counts"] == {"patched": 1}


def test_repair_loop_rejects_disallowed_file(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task_00001"
    write_task(task_dir)
    diagnosis = FailureDiagnosis(
        task_name="task_00001",
        failure_class="solution_error",
        confidence=0.9,
        evidence=["solve.sh exited 1"],
        allowed_repairs=["solution/solve.sh"],
    )

    def fake_llm(*args, **kwargs):
        return "```json\n" + json.dumps({"files": {"tests/test_outputs.py": "def test_bad(): pass"}}) + "\n```"

    run_repair_loop(
        tasks_dir=tasks_dir,
        diagnoses=[diagnosis],
        output_dir=tmp_path / "repair",
        max_attempts=1,
        validate=False,
        llm_call=fake_llm,
    )

    report = (tmp_path / "repair" / "repair_report.jsonl").read_text(encoding="utf-8")
    assert "repair tried to edit non-allowed file" in report


def test_repair_loop_accepts_validation_callback(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task_00001"
    write_task(task_dir)
    diagnosis = FailureDiagnosis(
        task_name="task_00001",
        failure_class="solution_error",
        confidence=0.9,
        evidence=["solve.sh exited 1"],
        allowed_repairs=["solution/solve.sh"],
    )

    def fake_llm(*args, **kwargs):
        return "```json\n" + json.dumps({"files": {"solution/solve.sh": "#!/bin/bash\ntrue\n"}, "notes": "fixed"}) + "\n```"

    def validate_task(task_dir: Path, validation_output: Path, timeout: int) -> dict:
        assert task_dir.name == "task_00001"
        assert validation_output.name == "validation"
        assert timeout == 123
        return {"status": "passed", "reward": 1.0}

    summary = run_repair_loop(
        tasks_dir=tasks_dir,
        diagnoses=[diagnosis],
        output_dir=tmp_path / "repair",
        max_attempts=1,
        validate=True,
        timeout=123,
        llm_call=fake_llm,
        validate_task=validate_task,
    )

    accepted = tmp_path / "repair" / "repaired_validated" / "task_00001" / "solution" / "solve.sh"
    assert accepted.read_text(encoding="utf-8") == "#!/bin/bash\ntrue\n"
    assert summary["status_counts"] == {"accepted": 1}
