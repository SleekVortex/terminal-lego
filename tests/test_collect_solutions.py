from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from scripts import collect_solutions as collector


def make_task(tasks_dir: Path, name: str, complete: bool = True) -> Path:
    task_dir = tasks_dir / name
    for relative in collector.REQUIRED_TASK_FILES:
        if not complete and relative == "tests/test_outputs.py":
            continue
        path = task_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    return task_dir


def write_trial(job_dir: Path, task_name: str, reward: float) -> None:
    trial_name = f"{task_name}__trial"
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "trial_name": trial_name,
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": None,
            }
        ),
        encoding="utf-8",
    )
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text("{}\n", encoding="utf-8")
    solve_path = trial_dir / "artifacts" / "logs" / "artifacts" / "solve.sh"
    solve_path.parent.mkdir(parents=True)
    solve_path.write_text("#!/bin/sh\n", encoding="utf-8")


def selected_task_names(command: Sequence[str]) -> list[str]:
    return [command[index + 1] for index, token in enumerate(command[:-1]) if token == "--include-task-name"]


def base_args(tasks_dir: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--tasks-dir",
        str(tasks_dir),
        "--output",
        str(output),
        "--target-accepted",
        "2",
        "--agent",
        "deepagent",
        "--model",
        "openai/test-model",
        "--api-base",
        "http://example.test/v1",
        "--api-key",
        "EMPTY",
        *extra,
    ]


def test_discover_tasks_uses_only_complete_task_directories(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    make_task(tasks_dir, "task_00002")
    make_task(tasks_dir, "task_00001")
    make_task(tasks_dir, "task_00003", complete=False)
    (tasks_dir / "notes").mkdir()

    assert collector.discover_task_names(tasks_dir) == ["task_00001", "task_00002"]
    assert collector.discover_task_names(tasks_dir, shuffle_seed=17) == collector.discover_task_names(
        tasks_dir,
        shuffle_seed=17,
    )


def test_collector_continues_after_failed_tasks_and_stops_at_exact_target(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    for index in range(5):
        make_task(tasks_dir, f"task_{index:05d}")

    output = tmp_path / "collection"
    args = collector.parse_args(base_args(tasks_dir, output, "--batch-size", "2"))
    accepted_names = {"task_00000", "task_00002"}
    scheduled_batches: list[list[str]] = []

    def fake_batch(command: Sequence[str], env: dict[str, str], log_path: Path) -> int:
        selected = selected_task_names(command)
        scheduled_batches.append(selected)
        job_dir = Path(command[2]) / env["JOB_NAME"]
        for task_name in selected:
            write_trial(job_dir, task_name, reward=1.0 if task_name in accepted_names else 0.0)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake batch\n", encoding="utf-8")
        return 0

    assert collector.run_collector(args, batch_runner=fake_batch) == 0

    summary = json.loads((output / "collection_summary.json").read_text(encoding="utf-8"))
    state = json.loads((output / "collector_state.json").read_text(encoding="utf-8"))
    accepted_rows = [
        json.loads(line) for line in (output / "accepted_trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed_rows = [
        json.loads(line) for line in (output / "failed_trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert scheduled_batches == [["task_00000", "task_00001"], ["task_00002"]]
    assert summary["status"] == "complete"
    assert summary["accepted_trajectories"] == 2
    assert summary["attempted_tasks"] == 3
    assert summary["failed_tasks"] == 1
    assert [row["task_name"] for row in accepted_rows] == ["task_00000", "task_00002"]
    assert [row["task_name"] for row in failed_rows] == ["task_00001"]
    assert state["attempted_tasks"] == ["task_00000", "task_00001", "task_00002"]

    def unexpected_batch(*_args, **_kwargs) -> int:
        raise AssertionError("a complete collection must not launch another batch")

    assert collector.run_collector(args, batch_runner=unexpected_batch) == 0


def test_collector_stops_after_consecutive_batches_without_rewards(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    for index in range(4):
        make_task(tasks_dir, f"task_{index:05d}")

    output = tmp_path / "collection"
    argv = base_args(
        tasks_dir,
        output,
        "--target-accepted",
        "1",
        "--batch-size",
        "1",
        "--max-stalled-batches",
        "2",
    )
    args = collector.parse_args(argv)
    calls = 0

    def empty_batch(_command: Sequence[str], _env: dict[str, str], log_path: Path) -> int:
        nonlocal calls
        calls += 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("endpoint unavailable\n", encoding="utf-8")
        return 1

    assert collector.run_collector(args, batch_runner=empty_batch) == 3
    summary = json.loads((output / "collection_summary.json").read_text(encoding="utf-8"))

    assert calls == 2
    assert summary["status"] == "stalled"
    assert summary["accepted_trajectories"] == 0
    assert summary["attempted_tasks"] == 2
    assert summary["failed_tasks"] == 2
    assert summary["consecutive_stalled_batches"] == 2


def test_parse_args_rejects_solver_options_managed_by_collector(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    with pytest.raises(SystemExit):
        collector.parse_args(base_args(tasks_dir, tmp_path / "output", "--", "--n-attempts", "2"))
