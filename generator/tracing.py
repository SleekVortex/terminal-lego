"""Crash-tolerant event traces for task-generation LLM calls."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


TRACE_SCHEMA_VERSION = "terminal-lego-task-generation-v1"


class GenerationTraceWriter:
    """Append one JSON event per line to a task-specific trace file."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._sequence_by_task: dict[str, int] = {}

    @staticmethod
    def _safe_task_name(task_name: str) -> str:
        safe_name = Path(task_name).name
        if safe_name != task_name or not safe_name:
            raise ValueError(f"Invalid task name for trace: {task_name!r}")
        return safe_name

    def record(self, task_name: str, event: str, **data: Any) -> None:
        safe_name = self._safe_task_name(task_name)
        with self._lock:
            sequence = self._sequence_by_task.get(safe_name, 0) + 1
            self._sequence_by_task[safe_name] = sequence
            row = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "run_id": self.run_id,
                "task_name": safe_name,
                "sequence": sequence,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": event,
                **data,
            }
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            with (self.output_dir / f"{safe_name}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line)

    def task_started(self, task_name: str, question: Any) -> None:
        self.record(
            task_name,
            "task_started",
            source={
                "question_id": question.question_id,
                "title": question.title,
                "tags": question.tags,
                "categories": question.categories,
                "selected_category": question.selected_category,
                "score": question.score,
                "link": question.link,
            },
        )

    def task_finished(self, task_name: str, success: bool, error: str | None = None) -> None:
        self.record(task_name, "task_finished", success=success, error=error)
