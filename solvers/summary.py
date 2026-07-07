"""Summarize solver job results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def reward_from_result(result: dict[str, Any]) -> float | None:
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


def exception_type_from_result(result: dict[str, Any]) -> str | None:
    exception_info = result.get("exception_info")
    if not exception_info:
        return None
    if isinstance(exception_info, dict):
        value = exception_info.get("exception_type") or exception_info.get("type")
        return str(value or "Exception")
    return type(exception_info).__name__


def classify_solution_trial(
    reward: float | None,
    reward_threshold: float,
    materialized_solution_exists: bool,
    exception_type: str | None,
) -> dict[str, Any]:
    if reward is not None and reward >= reward_threshold and materialized_solution_exists:
        accepted_with_timeout = exception_type == "AgentTimeoutError"
        return {
            "accepted": True,
            "accepted_with_timeout": accepted_with_timeout,
            "original_exception_type": exception_type if accepted_with_timeout else None,
            "rejection_reason": None,
        }
    if reward is not None and reward >= reward_threshold and not materialized_solution_exists:
        return {
            "accepted": False,
            "accepted_with_timeout": False,
            "original_exception_type": None,
            "rejection_reason": "reward_threshold_met_missing_solve_sh",
        }
    return {
        "accepted": False,
        "accepted_with_timeout": False,
        "original_exception_type": None,
        "rejection_reason": None,
    }


def iter_trial_results(job_dir: Path) -> Iterable[dict[str, Any]]:
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


def summarize_job(job_dir: Path, reward_threshold: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for item in iter_trial_results(job_dir):
        result_path: Path = item["result_path"]
        trial_dir: Path = item["trial_dir"]
        result: dict[str, Any] = item["result"]
        reward = reward_from_result(result)
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        materialized_solution_path = trial_dir / "artifacts" / "logs" / "artifacts" / "solve.sh"
        exception_info = result.get("exception_info")
        exception_type = exception_type_from_result(result)
        materialized_solution_exists = materialized_solution_path.exists()
        classification = classify_solution_trial(
            reward,
            reward_threshold,
            materialized_solution_exists,
            exception_type,
        )
        row = {
            "task_name": result.get("task_name"),
            "trial_name": result.get("trial_name"),
            "reward": reward,
            "accepted": classification["accepted"],
            "accepted_with_timeout": classification["accepted_with_timeout"],
            "original_exception_type": classification["original_exception_type"],
            "rejection_reason": classification["rejection_reason"],
            "exception_type": exception_type,
            "exception_info": exception_info,
            "result_path": str(result_path),
            "trajectory_path": str(trajectory_path) if trajectory_path.exists() else None,
            "materialized_solution_path": str(materialized_solution_path) if materialized_solution_exists else None,
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
