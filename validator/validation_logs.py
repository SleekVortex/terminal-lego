from __future__ import annotations

import json
from pathlib import Path

from validator.docker_runner import CommandResult


class TaskLogWriter:
    def __init__(self, root_dir: Path, task_name: str):
        self.task_name = task_name
        self.log_dir = root_dir / "validation_logs" / task_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write_command(self, stage: str, result: CommandResult) -> None:
        (self.log_dir / f"{stage}.stdout").write_text(result.stdout or "", encoding="utf-8", errors="replace")
        (self.log_dir / f"{stage}.stderr").write_text(result.stderr or "", encoding="utf-8", errors="replace")
        (self.log_dir / f"{stage}.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_result(self, result: dict) -> None:
        (self.log_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    def relative_to(self, root: Path) -> str:
        return str(self.log_dir.relative_to(root))
