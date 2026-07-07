"""Build Harbor job configs from solver CLI arguments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from solvers.harbor_runner import import_harbor
from solvers.parsing import parse_env, parse_key_value
from solvers.settings import PREINSTALLED_OPENCODE_AGENT, import_path_for_agent


def is_single_task_dir(path: Path) -> bool:
    return (path / "task.toml").is_file()


def build_agent_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = parse_key_value(args.agent_kwarg)
    if args.agent in {"terminus", "terminus-2"}:
        kwargs.setdefault("parser_name", "json")
        kwargs.setdefault("enable_summarize", True)
        kwargs.setdefault("record_terminal_session", True)
    if args.api_base:
        kwargs.setdefault("api_base", args.api_base)
    if args.api_key:
        kwargs.setdefault("api_key", args.api_key)

    trajectory_config = {}
    if args.trajectory_raw_content:
        trajectory_config["raw_content"] = True
    if args.trajectory_linear_history:
        trajectory_config["linear_history"] = True
    if args.trajectory_config_json:
        trajectory_config.update(json.loads(args.trajectory_config_json))
    if trajectory_config:
        existing = kwargs.get("trajectory_config")
        if isinstance(existing, dict):
            existing.update(trajectory_config)
        else:
            kwargs["trajectory_config"] = trajectory_config
    return kwargs


def build_agent_config(args: argparse.Namespace, AgentConfig, model_name: str | None):
    kwargs = {
        "model_name": model_name,
        "kwargs": build_agent_kwargs(args),
        "env": parse_env(args.agent_env),
        "include_logs": args.agent_include_logs or [],
        "exclude_logs": args.agent_exclude_logs or [],
    }
    if args.agent == PREINSTALLED_OPENCODE_AGENT:
        return AgentConfig(
            import_path=import_path_for_agent(args.agent),
            **kwargs,
        )
    return AgentConfig(
        name=args.agent,
        **kwargs,
    )


def build_job_config(args: argparse.Namespace):
    harbor = import_harbor()
    JobConfig = harbor["JobConfig"]
    RetryConfig = harbor["RetryConfig"]
    DatasetConfig = harbor["DatasetConfig"]
    TaskConfig = harbor["TaskConfig"]
    AgentConfig = harbor["AgentConfig"]
    EnvironmentConfig = harbor["EnvironmentConfig"]
    VerifierConfig = harbor["VerifierConfig"]

    tasks_dir = args.tasks_dir.resolve()
    if not tasks_dir.exists():
        raise SystemExit(f"Task path does not exist: {tasks_dir}")

    model_name = args.model or os.environ.get("MODEL_NAME")
    agent = build_agent_config(args, AgentConfig, model_name)

    environment = EnvironmentConfig(
        type=args.env,
        force_build=args.force_build,
        delete=args.delete,
        extra_docker_compose=[p.resolve() for p in args.extra_docker_compose],
        kwargs=parse_key_value(args.environment_kwarg),
        env=parse_env(args.environment_env),
    )

    verifier = VerifierConfig(
        disable=args.disable_verification,
        env=parse_env(args.verifier_env),
        include_logs=args.verifier_include_logs or [],
        exclude_logs=args.verifier_exclude_logs or [],
    )

    datasets = []
    tasks = []
    if is_single_task_dir(tasks_dir):
        tasks.append(TaskConfig(path=tasks_dir))
    else:
        datasets.append(
            DatasetConfig(
                path=tasks_dir,
                task_names=args.include_task_name or None,
                exclude_task_names=args.exclude_task_name or None,
                n_tasks=args.n_tasks,
            )
        )

    retry = RetryConfig(max_retries=args.max_retries)

    return JobConfig(
        job_name=args.job_name,
        jobs_dir=args.jobs_dir.resolve(),
        n_attempts=args.n_attempts,
        timeout_multiplier=args.timeout_multiplier,
        agent_timeout_multiplier=args.agent_timeout_multiplier,
        verifier_timeout_multiplier=args.verifier_timeout_multiplier,
        environment_build_timeout_multiplier=args.environment_build_timeout_multiplier,
        debug=args.debug,
        quiet=args.quiet,
        n_concurrent_trials=args.n_concurrent,
        retry=retry,
        environment=environment,
        verifier=verifier,
        agents=[agent],
        datasets=datasets,
        tasks=tasks,
        artifacts=args.artifact or [],
        extra_instruction_paths=[p.resolve() for p in args.extra_instruction_path],
    )
