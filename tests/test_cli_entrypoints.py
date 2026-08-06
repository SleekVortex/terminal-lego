from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINTS = [
    "generator.task_generator",
    "validator.validate_tasks",
    "sources.stackoverflow.prepare_dataset",
    "scripts.run_task_pipeline",
]


def test_cli_entrypoints_show_help() -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for module in ENTRYPOINTS:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"{module}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
