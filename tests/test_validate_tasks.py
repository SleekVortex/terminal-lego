from __future__ import annotations

import json
import subprocess
from pathlib import Path

from validator import validate_tasks as vt
from validator.docker_runner import CommandResult, DockerRunner, docker_safe_name


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


def ok(cmd: list[str] | None = None, stdout: str = "", stderr: str = "") -> CommandResult:
    return CommandResult(cmd or [], 0, stdout=stdout, stderr=stderr)


def fail(cmd: list[str] | None = None, stderr: str = "boom") -> CommandResult:
    return CommandResult(cmd or [], 1, stderr=stderr)


def test_atomic_counter_increment() -> None:
    counter = vt.AtomicCounter()
    assert counter.increment() == 1
    assert counter.increment() == 2


def test_docker_safe_name_replaces_invalid_repository_chars() -> None:
    assert docker_safe_name("task_00000") == "task-00000"
    assert docker_safe_name("Task__With bad/chars") == "task-with-bad-chars"
    assert docker_safe_name("___") == "task"


def test_command_result_from_completed_and_timeout() -> None:
    completed = subprocess.CompletedProcess(["x"], 2, stdout="out", stderr="err")
    assert CommandResult.from_completed(completed).to_dict()["stderr"] == "err"
    timeout = subprocess.TimeoutExpired(["x"], timeout=3, output="out", stderr="err")
    timed_out = CommandResult.from_timeout(["x"], timeout, timeout=3)
    assert timed_out.returncode == "timeout"
    assert timed_out.timed_out is True


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


def test_validate_task_passes_copies_task_and_writes_logs(monkeypatch, tmp_path: Path) -> None:
    reset_counters()
    task = make_task(tmp_path)
    output = tmp_path / "validated"
    output.mkdir()

    monkeypatch.setattr(DockerRunner, "build", lambda self, dockerfile, env_dir, timeout: ok(["build"]))
    monkeypatch.setattr(DockerRunner, "start", lambda self: ok(["run"]))
    monkeypatch.setattr(DockerRunner, "exec", lambda self, cmd, timeout=300, workdir=None: ok(["exec"] + cmd))
    monkeypatch.setattr(DockerRunner, "copy_to_container", lambda self, source, destination, timeout=10: ok(["cp", source, destination]))
    monkeypatch.setattr(DockerRunner, "read_reward", lambda self: ok(["reward"], stdout="1\n"))
    monkeypatch.setattr(DockerRunner, "cleanup", lambda self: None)

    result = vt.validate_task(task, output, timeout=10, total=1)

    assert result["status"] == "passed"
    assert result["reward"] == 1.0
    assert (output / "task_00000" / "solution" / "solve.sh").exists()
    assert (output / "validation_logs" / "task_00000" / "build.stdout").exists()
    result_json = json.loads((output / "validation_logs" / "task_00000" / "result.json").read_text())
    assert result_json["status"] == "passed"
    assert vt.passed.value == 1


def test_validate_task_handles_build_failure_with_logs(monkeypatch, tmp_path: Path) -> None:
    reset_counters()
    task = make_task(tmp_path)
    output = tmp_path / "validated"
    output.mkdir()

    monkeypatch.setattr(DockerRunner, "build", lambda self, dockerfile, env_dir, timeout: fail(["build"], stderr="build boom"))
    monkeypatch.setattr(DockerRunner, "cleanup", lambda self: None)

    result = vt.validate_task(task, output, timeout=10, total=1)

    assert result["status"] == "build_failed"
    assert result["error"] == "build boom"
    assert vt.build_failed.value == 1
    assert (output / "validation_logs" / "task_00000" / "build.stderr").read_text() == "build boom"


def test_validate_task_handles_failed_reward(monkeypatch, tmp_path: Path) -> None:
    reset_counters()
    task = make_task(tmp_path)
    output = tmp_path / "validated"
    output.mkdir()

    monkeypatch.setattr(DockerRunner, "build", lambda self, dockerfile, env_dir, timeout: ok(["build"]))
    monkeypatch.setattr(DockerRunner, "start", lambda self: ok(["run"]))
    monkeypatch.setattr(DockerRunner, "exec", lambda self, cmd, timeout=300, workdir=None: ok(["exec"] + cmd))
    monkeypatch.setattr(DockerRunner, "copy_to_container", lambda self, source, destination, timeout=10: ok(["cp", source, destination]))
    monkeypatch.setattr(DockerRunner, "read_reward", lambda self: ok(["reward"], stdout="0\n"))
    monkeypatch.setattr(DockerRunner, "cleanup", lambda self: None)

    result = vt.validate_task(task, output, timeout=10, total=1)

    assert result["status"] == "failed"
    assert result["reward"] == 0.0
    assert not (output / "task_00000").exists()
    assert vt.failed.value == 1
