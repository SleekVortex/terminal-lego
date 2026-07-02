#!/usr/bin/env python3
"""Generate agent solution rollouts for Terminal-Lego tasks using Harbor.

This module intentionally delegates harness-specific execution to Harbor. The
same CLI can run Terminus, mini-swe-agent, OpenHands, Codex, SWE-Agent, and any
other Harbor-supported agent/environment combination.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    if value.startswith("{") or value.startswith("["):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in KEY=VALUE item: {item!r}")
        values[key] = parse_scalar(raw_value.strip())
    return values


def parse_env(items: Optional[Sequence[str]]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in KEY=VALUE item: {item!r}")
        values[key] = raw_value.strip()
    return values


def import_harbor():
    try:
        from harbor.job import Job
        from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            VerifierConfig,
        )
    except ImportError as exc:
        raise SystemExit(
            "Harbor is required for solution generation. Install dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc

    return {
        "Job": Job,
        "JobConfig": JobConfig,
        "RetryConfig": RetryConfig,
        "DatasetConfig": DatasetConfig,
        "TaskConfig": TaskConfig,
        "AgentConfig": AgentConfig,
        "EnvironmentConfig": EnvironmentConfig,
        "VerifierConfig": VerifierConfig,
    }


def is_single_task_dir(path: Path) -> bool:
    return (path / "task.toml").is_file()


def build_agent_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
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
    agent = AgentConfig(
        name=args.agent,
        model_name=model_name,
        kwargs=build_agent_kwargs(args),
        env=parse_env(args.agent_env),
        include_logs=args.agent_include_logs or [],
        exclude_logs=args.agent_exclude_logs or [],
    )

    environment = EnvironmentConfig(
        type=args.env,
        force_build=args.force_build,
        delete=args.delete,
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


def reward_from_result(result: Dict[str, Any]) -> Optional[float]:
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    value = rewards.get("reward")
    if value is None:
        return None
    try:
        reward = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(reward):
        return None
    return reward


def iter_trial_results(job_dir: Path) -> Iterable[Dict[str, Any]]:
    for result_path in sorted(job_dir.glob("*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        yield {
            "result_path": result_path,
            "trial_dir": result_path.parent,
            "result": result,
        }


def summarize_job(job_dir: Path, reward_threshold: float) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for item in iter_trial_results(job_dir):
        result_path: Path = item["result_path"]
        trial_dir: Path = item["trial_dir"]
        result: Dict[str, Any] = item["result"]
        reward = reward_from_result(result)
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        materialized_solution_path = trial_dir / "artifacts" / "logs" / "artifacts" / "solve.sh"
        exception_info = result.get("exception_info")
        row = {
            "task_name": result.get("task_name"),
            "trial_name": result.get("trial_name"),
            "reward": reward,
            "accepted": reward is not None and reward >= reward_threshold,
            "exception_info": exception_info,
            "result_path": str(result_path),
            "trajectory_path": str(trajectory_path) if trajectory_path.exists() else None,
            "materialized_solution_path": (
                str(materialized_solution_path)
                if materialized_solution_path.exists()
                else None
            ),
        }
        rows.append(row)
        if row["accepted"]:
            accepted.append(row)
        else:
            failed.append(row)

    summary = {
        "job_dir": str(job_dir),
        "total_trials": len(rows),
        "accepted_trials": len(accepted),
        "failed_trials": len(failed),
        "reward_threshold": reward_threshold,
        "trials": rows,
    }

    summary_path = job_dir / "solution_generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    accepted_path = job_dir / "accepted_trajectories.jsonl"
    with accepted_path.open("w", encoding="utf-8") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    failed_path = job_dir / "failed_trials.jsonl"
    with failed_path.open("w", encoding="utf-8") as f:
        for row in failed:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary["summary_path"] = str(summary_path)
    summary["accepted_trajectories_path"] = str(accepted_path)
    summary["failed_trials_path"] = str(failed_path)
    return summary


async def run_job(config):
    harbor = import_harbor()
    Job = harbor["Job"]
    job = await Job.create(config)
    return await job.run()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True, type=Path, help="Task or dataset directory")
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    parser.add_argument(
        "--job-name",
        default=f"solution-rollouts-{time.strftime('%Y%m%dT%H%M%S')}",
    )
    parser.add_argument("--agent", default=os.environ.get("AGENT", "terminus-2"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--env", default=os.environ.get("HARBOR_ENV", "docker"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--n-attempts", type=int, default=int(os.environ.get("N_ATTEMPTS", "1")))
    parser.add_argument("--n-concurrent", type=int, default=int(os.environ.get("N_CONCURRENT", "1")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("MAX_RETRIES", "0")))
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=None)
    parser.add_argument("--verifier-timeout-multiplier", type=float, default=None)
    parser.add_argument("--environment-build-timeout-multiplier", type=float, default=None)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--delete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--disable-verification", action="store_true")
    parser.add_argument("--reward-threshold", type=float, default=1.0)
    parser.add_argument("--extra-instruction-path", action="append", type=Path, default=[])
    parser.add_argument("--include-task-name", action="append", default=[])
    parser.add_argument("--exclude-task-name", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--agent-kwarg", action="append", default=[])
    parser.add_argument("--agent-env", action="append", default=[])
    parser.add_argument("--agent-include-logs", action="append", default=[])
    parser.add_argument("--agent-exclude-logs", action="append", default=[])
    parser.add_argument("--environment-kwarg", action="append", default=[])
    parser.add_argument("--environment-env", action="append", default=[])
    parser.add_argument("--verifier-env", action="append", default=[])
    parser.add_argument("--verifier-include-logs", action="append", default=[])
    parser.add_argument("--verifier-exclude-logs", action="append", default=[])
    parser.add_argument("--trajectory-raw-content", action="store_true")
    parser.add_argument("--trajectory-linear-history", action="store_true")
    parser.add_argument("--trajectory-config-json", default=None)
    parser.add_argument("--dry-run-config", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = build_job_config(args)

    if args.dry_run_config:
        args.dry_run_config.parent.mkdir(parents=True, exist_ok=True)
        args.dry_run_config.write_text(
            config.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote Harbor job config to {args.dry_run_config}")
        return 0

    asyncio.run(run_job(config))

    job_dir = config.jobs_dir / config.job_name
    summary = summarize_job(job_dir, args.reward_threshold)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
