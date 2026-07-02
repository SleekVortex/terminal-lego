from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from generator import solution_generator as sg


def test_parse_scalar_and_key_value_items() -> None:
    assert sg.parse_scalar("true") is True
    assert sg.parse_scalar("false") is False
    assert sg.parse_scalar("null") is None
    assert sg.parse_scalar("42") == 42
    assert sg.parse_scalar("3.5") == 3.5
    assert sg.parse_scalar('{"a": 1}') == {"a": 1}
    assert sg.parse_key_value(["a=1", "enabled=true", "name=glm"]) == {
        "a": 1,
        "enabled": True,
        "name": "glm",
    }
    assert sg.parse_env(["A=1", "EMPTY=", "FLAG=true"]) == {
        "A": "1",
        "EMPTY": "",
        "FLAG": "true",
    }

    with pytest.raises(ValueError, match="KEY=VALUE"):
        sg.parse_key_value(["broken"])
    with pytest.raises(ValueError, match="Empty key"):
        sg.parse_key_value([" =1"])


def test_build_agent_kwargs_adds_terminus_defaults_and_trajectory_config() -> None:
    args = argparse.Namespace(
        agent="terminus-2",
        agent_kwarg=["temperature=0.2", "parser_name=xml"],
        api_base="http://localhost:30002/v1",
        api_key="EMPTY",
        trajectory_raw_content=True,
        trajectory_linear_history=True,
        trajectory_config_json='{"max_tokens": 12}',
    )

    kwargs = sg.build_agent_kwargs(args)

    assert kwargs["parser_name"] == "xml"
    assert kwargs["enable_summarize"] is True
    assert kwargs["record_terminal_session"] is True
    assert kwargs["temperature"] == 0.2
    assert kwargs["api_base"] == "http://localhost:30002/v1"
    assert kwargs["api_key"] == "EMPTY"
    assert kwargs["trajectory_config"] == {
        "raw_content": True,
        "linear_history": True,
        "max_tokens": 12,
    }


def test_build_job_config_for_single_task_and_dataset(tmp_path: Path) -> None:
    single_task = tmp_path / "task_00000"
    single_task.mkdir()
    (single_task / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")

    args = sg.parse_args(
        [
            "--tasks-dir",
            str(single_task),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            "job-one",
            "--agent",
            "terminus-2",
            "--model",
            "openai/glm",
            "--api-base",
            "http://localhost:30002/v1",
            "--api-key",
            "EMPTY",
            "--agent-env",
            "A=1",
            "--environment-kwarg",
            "privileged=false",
            "--verifier-env",
            "B=2",
            "--extra-instruction-path",
            str(tmp_path / "extra.md"),
        ]
    )
    config = sg.build_job_config(args)
    dumped = config.model_dump()

    assert dumped["job_name"] == "job-one"
    assert dumped["agents"][0]["name"] == "terminus-2"
    assert dumped["agents"][0]["model_name"] == "openai/glm"
    assert dumped["agents"][0]["env"] == {"A": "1"}
    assert dumped["environment"]["kwargs"] == {"privileged": False}
    assert dumped["verifier"]["env"] == {"B": "2"}
    assert len(dumped["tasks"]) == 1
    assert not dumped["datasets"]

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    args = sg.parse_args(
        [
            "--tasks-dir",
            str(dataset),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            "job-dataset",
            "--include-task-name",
            "task_a",
            "--exclude-task-name",
            "task_b",
            "--n-tasks",
            "3",
        ]
    )
    dataset_config = sg.build_job_config(args).model_dump()
    assert len(dataset_config["datasets"]) == 1
    assert dataset_config["datasets"][0]["task_names"] == ["task_a"]
    assert dataset_config["datasets"][0]["exclude_task_names"] == ["task_b"]
    assert dataset_config["datasets"][0]["n_tasks"] == 3
    assert not dataset_config["tasks"]


def test_build_job_config_rejects_missing_task_path(tmp_path: Path) -> None:
    args = sg.parse_args(["--tasks-dir", str(tmp_path / "missing")])
    with pytest.raises(SystemExit, match="Task path does not exist"):
        sg.build_job_config(args)


def test_reward_result_iteration_and_summary_files(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    accepted = job_dir / "trial-a"
    failed = job_dir / "trial-b"
    bad = job_dir / "trial-bad"
    (accepted / "agent").mkdir(parents=True)
    (accepted / "artifacts" / "logs" / "artifacts").mkdir(parents=True)
    failed.mkdir(parents=True)
    bad.mkdir(parents=True)

    (accepted / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task_a",
                "trial_name": "trial-a",
                "verifier_result": {"rewards": {"reward": "1"}},
            }
        ),
        encoding="utf-8",
    )
    (accepted / "agent" / "trajectory.json").write_text("[]", encoding="utf-8")
    (accepted / "artifacts" / "logs" / "artifacts" / "solve.sh").write_text(
        "#!/bin/bash\n",
        encoding="utf-8",
    )
    (failed / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task_b",
                "trial_name": "trial-b",
                "verifier_result": {"rewards": {"reward": 0}},
                "exception_info": {"type": "AgentTimeoutError"},
            }
        ),
        encoding="utf-8",
    )
    (bad / "result.json").write_text("{bad json", encoding="utf-8")

    assert sg.reward_from_result({"verifier_result": {"rewards": {"reward": "0.5"}}}) == 0.5
    assert sg.reward_from_result({"verifier_result": {"rewards": {"reward": "nan"}}}) is None

    results = list(sg.iter_trial_results(job_dir))
    assert [item["trial_dir"].name for item in results] == ["trial-a", "trial-b"]

    summary = sg.summarize_job(job_dir, reward_threshold=1.0)

    assert summary["total_trials"] == 2
    assert summary["accepted_trials"] == 1
    assert summary["failed_trials"] == 1
    assert Path(summary["summary_path"]).exists()
    assert Path(summary["accepted_trajectories_path"]).read_text(encoding="utf-8").count("\n") == 1
    assert Path(summary["failed_trials_path"]).read_text(encoding="utf-8").count("\n") == 1


def test_parse_args_exposes_dry_run_and_harbor_options(tmp_path: Path) -> None:
    args = sg.parse_args(
        [
            "--tasks-dir",
            str(tmp_path),
            "--dry-run-config",
            str(tmp_path / "config.json"),
            "--disable-verification",
            "--no-delete",
            "--agent-include-logs",
            "agent.log",
        ]
    )

    assert args.dry_run_config == tmp_path / "config.json"
    assert args.disable_verification is True
    assert args.delete is False
    assert args.agent_include_logs == ["agent.log"]
