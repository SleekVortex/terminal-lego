from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from threading import Event

from scripts import run_task_pipeline as pipeline


def test_pipeline_validates_each_task_before_generation_finishes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = pipeline.build_paths(tmp_path / "run")
    for path in (paths["seed"].parent, paths["logs"], paths["state"]):
        path.mkdir(parents=True, exist_ok=True)
    paths["seed"].write_text(
        json.dumps(
            {
                "questions": [
                    {"question_id": 1, "accepted_answer": {"body": "a"}},
                    {"question_id": 2, "accepted_answer": {"body": "b"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    validation_started = Event()

    class FakeValidator:
        def __init__(self, output_dir: Path, timeout: int) -> None:
            self.output_dir = output_dir

        def validate(self, task_dir: Path, total: int) -> dict:
            assert total == 2
            validation_started.set()
            return {"task": task_dir.name, "status": "passed", "reward": 1}

    def fake_run_generation(*, output_dir: Path, on_generated, **kwargs) -> dict:
        first = output_dir / "task_00000"
        second = output_dir / "task_00001"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        on_generated(first, 2)
        assert validation_started.wait(timeout=2)
        on_generated(second, 2)
        return {"total_processed": 2, "success_count": 2, "elapsed_seconds": 0.1}

    monkeypatch.setattr(pipeline, "TaskValidator", FakeValidator)
    monkeypatch.setattr(pipeline.task_generator, "configure_generation", lambda **kwargs: None)
    monkeypatch.setattr(pipeline.task_generator, "run_generation", fake_run_generation)

    marker = pipeline.run_streaming_generation_and_validation(
        Namespace(
            resume=False,
            api_base="http://example.test/v1",
            api_key="EMPTY",
            model="test-model",
            timeout=1,
            generate_workers=2,
            validate_workers=2,
        ),
        paths,
    )

    report = json.loads((paths["accepted"] / "validation_report.json").read_text())
    assert marker["status"] == "completed"
    assert report["total"] == 2
    assert report["passed"] == 2
