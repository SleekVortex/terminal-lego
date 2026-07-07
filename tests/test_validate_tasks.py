from __future__ import annotations

import subprocess
from pathlib import Path

from validator import validate_tasks as vt


def make_task(tmp_path: Path, name: str = "task_00000") -> Path:
    task_dir = tmp_path / name
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n", encoding="utf-8")
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    return task_dir


def reset_counters() -> None:
    for counter in (vt.progress, vt.passed, vt.failed, vt.build_failed, vt.error_count):
        counter.value = 0


def completed(cmd, code: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr=stderr)


def test_atomic_counter_increment() -> None:
    counter = vt.AtomicCounter()
    assert counter.increment() == 1
    assert counter.increment() == 2


def test_docker_safe_name_replaces_invalid_repository_chars() -> None:
    assert vt.docker_safe_name("task_00000") == "task-00000"
    assert vt.docker_safe_name("Task__With bad/chars") == "task-with-bad-chars"
    assert vt.docker_safe_name("___") == "task"


def test_validate_task_reports_missing_required_files(tmp_path: Path) -> None:
    output = tmp_path / "validated"
    task = tmp_path / "task_00000"
    task.mkdir()

    assert vt.validate_task(task, output, timeout=1, total=1)["status"] == "missing_dockerfile"

    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu\n", encoding="utf-8")
    assert vt.validate_task(task, output, timeout=1, total=1)["status"] == "missing_solution"

    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    assert vt.validate_task(task, output, timeout=1, total=1)["status"] == "missing_tests"


def test_validate_task_passes_and_copies_task_with_mocked_docker(monkeypatch, tmp_path: Path) -> None:
    reset_counters()
    task = make_task(tmp_path)
    output = tmp_path / "validated"
    output.mkdir()
    commands = []

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        commands.append(cmd)
        if cmd[:3] == ["docker", "exec", "tl-val-task-00000-" + str(vt.os.getpid())] and cmd[-1] == "/logs/verifier/reward.txt":
            return completed(cmd, stdout="1\n")
        return completed(cmd)

    monkeypatch.setattr(vt.subprocess, "run", fake_run)

    result = vt.validate_task(task, output, timeout=10, total=1)

    assert result["status"] == "passed"
    assert result["reward"] == 1.0
    assert (output / "task_00000" / "task.toml").exists() is False
    assert (output / "task_00000" / "solution" / "solve.sh").exists()
    assert vt.passed.value == 1
    assert any(cmd[:2] == ["docker", "build"] for cmd in commands)
    assert any(cmd[:4] == ["docker", "build", "-t", "tl-validate-task-00000"] for cmd in commands)
    assert any(cmd[:2] == ["docker", "rm"] for cmd in commands)


def test_validate_task_handles_build_failure_with_mocked_docker(monkeypatch, tmp_path: Path) -> None:
    reset_counters()
    task = make_task(tmp_path)
    output = tmp_path / "validated"
    output.mkdir()

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        if cmd[:2] == ["docker", "build"]:
            return completed(cmd, code=1, stderr="build boom")
        return completed(cmd)

    monkeypatch.setattr(vt.subprocess, "run", fake_run)

    result = vt.validate_task(task, output, timeout=10, total=1)

    assert result["status"] == "build_failed"
    assert result["error"] == "build boom"
    assert vt.build_failed.value == 1


def test_validate_task_handles_failed_reward(monkeypatch, tmp_path: Path) -> None:
    reset_counters()
    task = make_task(tmp_path)
    output = tmp_path / "validated"
    output.mkdir()

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        if cmd[:2] == ["docker", "exec"] and cmd[-1] == "/logs/verifier/reward.txt":
            return completed(cmd, stdout="0\n")
        return completed(cmd)

    monkeypatch.setattr(vt.subprocess, "run", fake_run)

    result = vt.validate_task(task, output, timeout=10, total=1)

    assert result["status"] == "failed"
    assert result["reward"] == 0.0
    assert not (output / "task_00000").exists()
    assert vt.failed.value == 1
