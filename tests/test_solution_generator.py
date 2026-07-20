from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import pytest

from solvers import config_builder
from solvers import parsing
from solvers import run_solutions as sg
from solvers import settings as solver_settings
from solvers import summary as solution_summary
from solvers.agents.deepagent import (
    DEEPAGENT_PROJECT_DIR,
    DeepAgent,
    merge_reasoning_records,
    trajectory_from_langgraph_result,
)
from solvers.agents.preinstalled_opencode import (
    OPENCODE_SYSTEM_PROMPT,
    PreinstalledOpenCode,
)


def test_parse_scalar_and_key_value_items() -> None:
    assert parsing.parse_scalar("true") is True
    assert parsing.parse_scalar("false") is False
    assert parsing.parse_scalar("null") is None
    assert parsing.parse_scalar("42") == 42
    assert parsing.parse_scalar("3.5") == 3.5
    assert parsing.parse_scalar('{"a": 1}') == {"a": 1}
    assert parsing.parse_key_value(["a=1", "enabled=true", "name=glm"]) == {
        "a": 1,
        "enabled": True,
        "name": "glm",
    }
    assert parsing.parse_env(["A=1", "EMPTY=", "FLAG=true"]) == {
        "A": "1",
        "EMPTY": "",
        "FLAG": "true",
    }

    with pytest.raises(ValueError, match="KEY=VALUE"):
        parsing.parse_key_value(["broken"])
    with pytest.raises(ValueError, match="Empty key"):
        parsing.parse_key_value([" =1"])


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

    kwargs = config_builder.build_agent_kwargs(args)

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
            "--extra-docker-compose",
            str(tmp_path / "bridge.yaml"),
            "--environment-kwarg",
            "privileged=false",
            "--verifier-env",
            "B=2",
            "--extra-instruction-path",
            str(tmp_path / "extra.md"),
        ]
    )
    config = config_builder.build_job_config(args)
    dumped = config.model_dump()

    assert dumped["job_name"] == "job-one"
    assert dumped["agents"][0]["name"] == "terminus-2"
    assert dumped["agents"][0]["model_name"] == "openai/glm"
    assert dumped["agents"][0]["env"] == {"A": "1"}
    assert dumped["environment"]["extra_docker_compose"] == [tmp_path / "bridge.yaml"]
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
    dataset_config = config_builder.build_job_config(args).model_dump()
    assert len(dataset_config["datasets"]) == 1
    assert dataset_config["datasets"][0]["task_names"] == ["task_a"]
    assert dataset_config["datasets"][0]["exclude_task_names"] == ["task_b"]
    assert dataset_config["datasets"][0]["n_tasks"] == 3
    assert not dataset_config["tasks"]


def test_build_job_config_uses_import_path_for_preinstalled_opencode(
    tmp_path: Path,
) -> None:
    single_task = tmp_path / "task_00000"
    single_task.mkdir()
    (single_task / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")

    args = sg.parse_args(
        [
            "--tasks-dir",
            str(single_task),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--agent",
            "preinstalled-opencode",
            "--model",
            "openai/glm",
            "--api-base",
            "http://host.docker.internal:30003/v1",
            "--api-key",
            "EMPTY",
        ]
    )

    config = config_builder.build_job_config(args)
    agent = config.agents[0]

    assert agent.name is None
    assert agent.import_path == solver_settings.import_path_for_agent(
        solver_settings.PREINSTALLED_OPENCODE_AGENT
    )
    assert agent.model_name == "openai/glm"
    assert agent.kwargs["api_base"] == "http://host.docker.internal:30003/v1"
    assert agent.kwargs["api_key"] == "EMPTY"


def test_build_job_config_uses_import_path_for_deepagent(tmp_path: Path) -> None:
    single_task = tmp_path / "task_00000"
    single_task.mkdir()
    (single_task / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")

    args = sg.parse_args(
        [
            "--tasks-dir",
            str(single_task),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--agent",
            "deepagent",
            "--model",
            "openai/glm-5.2-fp8",
            "--api-base",
            "http://host.docker.internal:30301/v1",
            "--api-key",
            "EMPTY",
        ]
    )

    config = config_builder.build_job_config(args)
    agent = config.agents[0]

    assert agent.name is None
    assert agent.import_path == solver_settings.import_path_for_agent(
        solver_settings.DEEPAGENT_AGENT
    )
    assert agent.model_name == "openai/glm-5.2-fp8"
    assert agent.kwargs["api_base"] == "http://host.docker.internal:30301/v1"
    assert agent.kwargs["api_key"] == "EMPTY"


def test_deepagent_configures_local_chat_completions(tmp_path: Path) -> None:
    agent = DeepAgent(
        logs_dir=tmp_path,
        model_name="openai/glm-5.2-fp8",
        api_base="http://host.docker.internal:30301/v1",
        api_key="EMPTY",
        model_kwargs={"timeout": 1800},
    )

    assert agent.project_path == DEEPAGENT_PROJECT_DIR
    assert agent.graph == "deepagent"
    assert agent.model_kwargs == {
        "timeout": 1800,
        "base_url": "http://host.docker.internal:30301/v1",
        "use_responses_api": False,
        "api_key": "EMPTY",
    }
    assert agent.configurable == {"cwd": "/app"}


def test_deepagent_project_pins_prerelease_before_code_package() -> None:
    project_config = json.loads(
        (DEEPAGENT_PROJECT_DIR / "langgraph.json").read_text(encoding="utf-8")
    )

    dependencies = project_config["dependencies"]
    assert "deepagents-0.7.0a7-py3-none-any.whl#sha256=" in dependencies[0]
    assert dependencies[1] == "deepagents-code==0.1.43"


def test_deepagent_converts_langgraph_messages_to_atif() -> None:
    result = {
            "messages": [
                {"type": "human", "content": "fix it"},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "shell",
                            "args": {"command": "pwd"},
                        }
                    ],
                    "usage_metadata": {
                        "input_tokens": 11,
                        "output_tokens": 3,
                        "input_token_details": {"cache_read": 2},
                    },
                },
                {"type": "tool", "content": "/app"},
                {
                    "type": "ai",
                    "content": "done",
                    "usage_metadata": {"input_tokens": 17, "output_tokens": 4},
                },
            ]
        }
    merged = merge_reasoning_records(
        result,
        {
            "records": [
                {
                    "message_index": 1,
                    "reasoning_content": "I should inspect the working directory.",
                    "reasoning_tokens": 9,
                },
                {
                    "message_index": 3,
                    "reasoning_content": "The task is complete.",
                    "reasoning_tokens": 5,
                },
            ]
        },
    )
    trajectory = trajectory_from_langgraph_result(
        result,
        instruction="fix it",
        model_name="openai/glm-5.2-fp8",
        session_id="session-1",
    )

    assert merged == 2
    assert [step.source for step in trajectory.steps] == [
        "system",
        "user",
        "agent",
        "agent",
    ]
    tool_step = trajectory.steps[2]
    assert tool_step.tool_calls is not None
    assert tool_step.tool_calls[0].function_name == "shell"
    assert tool_step.observation is not None
    assert tool_step.observation.results[0].source_call_id == "call-1"
    assert tool_step.observation.results[0].content == "/app"
    assert tool_step.reasoning_content == "I should inspect the working directory."
    assert tool_step.metrics is not None
    assert tool_step.metrics.extra == {"reasoning_tokens": 9}
    assert trajectory.steps[3].reasoning_content == "The task is complete."
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 28
    assert trajectory.final_metrics.total_completion_tokens == 7
    assert trajectory.final_metrics.total_cached_tokens == 2
    assert trajectory.final_metrics.extra == {"total_reasoning_tokens": 14}


def test_build_job_config_rejects_missing_task_path(tmp_path: Path) -> None:
    args = sg.parse_args(["--tasks-dir", str(tmp_path / "missing")])
    with pytest.raises(SystemExit, match="Task path does not exist"):
        config_builder.build_job_config(args)


def test_reward_result_iteration_and_summary_files(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    accepted = job_dir / "trial-a"
    accepted_timeout = job_dir / "trial-timeout"
    failed_missing_solution = job_dir / "trial-missing-solve-sh"
    failed = job_dir / "trial-b"
    bad = job_dir / "trial-bad"
    (accepted / "agent").mkdir(parents=True)
    (accepted / "artifacts" / "logs" / "artifacts").mkdir(parents=True)
    (accepted_timeout / "artifacts" / "logs" / "artifacts").mkdir(parents=True)
    failed_missing_solution.mkdir(parents=True)
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
    (accepted_timeout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task_timeout",
                "trial_name": "trial-timeout",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "exception_info": {"exception_type": "AgentTimeoutError"},
            }
        ),
        encoding="utf-8",
    )
    (
        accepted_timeout
        / "artifacts"
        / "logs"
        / "artifacts"
        / "solve.sh"
    ).write_text("#!/bin/bash\n", encoding="utf-8")
    (failed_missing_solution / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task_missing_solution",
                "trial_name": "trial-missing-solve-sh",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
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

    assert solution_summary.reward_from_result({"verifier_result": {"rewards": {"reward": "0.5"}}}) == 0.5
    assert solution_summary.reward_from_result({"verifier_result": {"rewards": {"reward": "nan"}}}) is None
    assert solution_summary.exception_type_from_result({"exception_info": {"type": "AgentTimeoutError"}}) == "AgentTimeoutError"

    results = list(solution_summary.iter_trial_results(job_dir))
    assert [item["trial_dir"].name for item in results] == [
        "trial-a",
        "trial-b",
        "trial-missing-solve-sh",
        "trial-timeout",
    ]

    summary = solution_summary.summarize_job(job_dir, reward_threshold=1.0)

    assert summary["total_trials"] == 4
    assert summary["accepted_trials"] == 2
    assert summary["failed_trials"] == 2
    assert Path(summary["summary_path"]).exists()
    accepted_rows = [
        json.loads(line)
        for line in Path(summary["accepted_trajectories_path"]).read_text(encoding="utf-8").splitlines()
    ]
    failed_rows = [
        json.loads(line)
        for line in Path(summary["failed_trials_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(accepted_rows) == 2
    assert len(failed_rows) == 2
    timeout_row = next(row for row in accepted_rows if row["task_name"] == "task_timeout")
    assert timeout_row["accepted_with_timeout"] is True
    assert timeout_row["original_exception_type"] == "AgentTimeoutError"
    missing_solution_row = next(
        row for row in failed_rows if row["task_name"] == "task_missing_solution"
    )
    assert missing_solution_row["rejection_reason"] == "reward_threshold_met_missing_solve_sh"


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
            "--extra-docker-compose",
            str(tmp_path / "bridge.yaml"),
        ]
    )

    assert args.dry_run_config == tmp_path / "config.json"
    assert args.disable_verification is True
    assert args.delete is False
    assert args.agent_include_logs == ["agent.log"]
    assert args.extra_docker_compose == [tmp_path / "bridge.yaml"]


def test_preinstalled_opencode_install_does_not_download_runtime() -> None:
    assert PreinstalledOpenCode.name() == "preinstalled-opencode"
    assert PreinstalledOpenCode.get_version_command(None) == "opencode --version"

    source = inspect.getsource(PreinstalledOpenCode.install)
    assert "curl" not in source
    assert "npm" not in source
    assert "nvm install" not in source
    assert "/opt/terminal-lego/opencode/bin/opencode" in source


def test_preinstalled_opencode_registers_glm_compatible_defaults() -> None:
    defaults = PreinstalledOpenCode._DEFAULT_CONFIG

    assert defaults["agent"]["title"]["disable"] is True


def test_preinstalled_opencode_registers_glm_custom_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://host.docker.internal:30003/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")

    agent = PreinstalledOpenCode(
        model_name="openai/glm-5.2-fp8",
        logs_dir=tmp_path,
    )
    command = agent._build_register_config_command()

    assert agent.model_name == "glm/glm-5.2-fp8"
    assert command is not None
    assert '"model": "glm/glm-5.2-fp8"' in command
    assert '"small_model": "glm/glm-5.2-fp8"' in command
    assert '"glm": {' in command
    assert '"npm": "@ai-sdk/openai-compatible"' in command
    assert '"baseURL": "http://host.docker.internal:30003/v1"' in command
    assert '"prompt":' in command
    assert "Harbor/Terminal-Lego task container" in command
    assert '"tool_call": true' in command
    assert '"interleaved": {' in command
    assert '"field": "reasoning_content"' in command


def test_preinstalled_opencode_adds_system_prompt_to_trajectory(
    tmp_path: Path,
) -> None:
    agent = PreinstalledOpenCode(
        model_name="openai/glm-5.2-fp8",
        logs_dir=tmp_path,
    )
    agent._instruction = "solve the task"

    trajectory = agent._convert_events_to_trajectory(
        [
            {
                "type": "step_start",
                "timestamp": 1000,
                "sessionID": "session-1",
            },
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "done",
                },
            },
            {
                "type": "step_finish",
                "part": {
                    "tokens": {
                        "input": 10,
                        "output": 2,
                    },
                },
            },
        ]
    )

    assert trajectory is not None
    assert [step.source for step in trajectory.steps] == ["system", "user", "agent"]
    assert [step.step_id for step in trajectory.steps] == [1, 2, 3]
    assert trajectory.steps[0].message == OPENCODE_SYSTEM_PROMPT
    assert trajectory.steps[0].extra == {
        "origin": "terminal-lego.preinstalled-opencode",
        "prompt_config": "agent.build.prompt",
    }
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_steps == 3
