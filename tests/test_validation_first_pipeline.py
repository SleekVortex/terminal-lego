from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validation_first_pipeline.py"
spec = importlib.util.spec_from_file_location("validation_first_pipeline", SCRIPT_PATH)
vfp = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(vfp)


def write_text(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_complete_task(root: Path, name: str) -> None:
    write_text(root / name / "instruction.md")
    write_text(root / name / "environment" / "Dockerfile", "FROM python:3.12-slim-bookworm\n")
    write_text(root / name / "solution" / "solve.sh", "#!/bin/bash\ntrue\n")
    write_text(root / name / "tests" / "test_outputs.py", "def test_ok():\n    assert True\n")
    write_text(root / name / "tests" / "test.sh", "#!/bin/bash\npytest /tests/test_outputs.py\n")


def test_collect_failures_marks_regen_eligibility(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    make_complete_task(tasks_dir, "task_00002")
    write_text(tasks_dir / "task_00003" / "instruction.md")
    validation_report = tmp_path / "validation_report.json"
    validation_report.write_text(
        json.dumps(
            {
                "results": [
                    {"task": "task_00001", "status": "passed", "reward": 1.0},
                    {"task": "task_00002", "status": "failed", "reward": 0.0, "error": "pytest failed"},
                    {"task": "task_00003", "status": "failed", "reward": 0.0, "error": "missing solve.sh"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "failed_tasks.jsonl"

    vfp.collect_failures(
        argparse.Namespace(
            validation_report=validation_report,
            tasks_dir=tasks_dir,
            output=output,
        )
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["task"] for row in rows] == ["task_00002", "task_00003"]
    assert rows[0]["regen_eligible"] is True
    assert rows[0]["final_status"] == "needs_dockerfile_regen"
    assert rows[1]["regen_eligible"] is False
    assert rows[1]["final_status"] == "discard_missing_artifacts"

    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["failed_total"] == 2
    assert summary["regen_eligible"] == 1
    assert summary["discard_missing_artifacts"] == 1


def test_merge_accepted_copies_baseline_and_regenerated_tasks(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline_validated"
    regenerated = tmp_path / "regenerated_validated"
    make_complete_task(baseline, "task_00001")
    make_complete_task(baseline, "task_00002")
    make_complete_task(regenerated, "task_00002")
    make_complete_task(regenerated, "task_00003")
    output_dir = tmp_path / "final_accepted"

    vfp.merge_accepted(
        argparse.Namespace(
            baseline_validated_dir=baseline,
            regenerated_validated_dir=regenerated,
            output_dir=output_dir,
        )
    )

    assert sorted(path.name for path in output_dir.iterdir() if path.is_dir()) == [
        "task_00001",
        "task_00002",
        "task_00003",
    ]
    report = json.loads((output_dir / "final_accepted_report.json").read_text(encoding="utf-8"))
    assert report["final_accepted"] == 3
    assert report["duplicates"] == 1
    duplicate_rows = [row for row in report["rows"] if row["duplicate"]]
    assert duplicate_rows == [{"task": "task_00002", "source": "regenerated", "copied": True, "duplicate": True}]
