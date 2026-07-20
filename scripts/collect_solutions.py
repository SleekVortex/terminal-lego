#!/usr/bin/env python3
"""Collect accepted solution trajectories until a requested quota is reached.

The collector runs the existing ``generate_solutions.sh`` wrapper in bounded
batches. Tasks scheduled once are never scheduled again, failed trials do not
count toward the quota, and all progress is persisted so the command can be
restarted with the same arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from solvers.summary import summarize_job


REPO_ROOT = Path(__file__).resolve().parents[1]
SOLUTION_WRAPPER = REPO_ROOT / "scripts" / "generate_solutions.sh"
STATE_SCHEMA_VERSION = "terminal-lego-solution-collector-v1"
REQUIRED_TASK_FILES = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
    "tests/test_outputs.py",
)
FORBIDDEN_SOLVER_OPTIONS = {
    "--agent",
    "--api-base",
    "--api-key",
    "--disable-verification",
    "--dry-run-config",
    "--exclude-task-name",
    "--include-task-name",
    "--job-name",
    "--jobs-dir",
    "--model",
    "--n-attempts",
    "--n-concurrent",
    "--n-tasks",
    "--reward-threshold",
    "--tasks-dir",
}


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True, type=Path, help="Directory with complete task_* directories.")
    parser.add_argument("--output", required=True, type=Path, help="Persistent collector output directory.")
    parser.add_argument(
        "--target-accepted",
        "--target",
        dest="target_accepted",
        required=True,
        type=int,
        help="Number of unique accepted trajectories to collect.",
    )
    parser.add_argument("--agent", default=os.environ.get("AGENT", "terminus-2"), help="Harbor agent/harness.")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"), help="Model name passed to the harness.")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"), help="OpenAI-compatible API base.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--env", default=os.environ.get("HARBOR_ENV", "docker"), help="Harbor environment.")
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=int(os.environ.get("N_CONCURRENT", "1")),
        help="Concurrent Harbor trials per batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tasks scheduled per batch. Default: four times --n-concurrent.",
    )
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("MAX_RETRIES", "0")))
    parser.add_argument("--reward-threshold", type=float, default=1.0)
    parser.add_argument(
        "--max-stalled-batches",
        type=int,
        default=3,
        help="Stop after this many consecutive batches without a numeric verifier reward.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Deterministically shuffle task order with this seed.",
    )
    parser.add_argument(
        "--docker-network-strategy",
        choices=("bridge", "compose"),
        default=os.environ.get("DOCKER_NETWORK_STRATEGY", "bridge"),
    )
    parser.add_argument(
        "--cleanup-docker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let generate_solutions.sh clean batch Docker resources.",
    )
    parser.add_argument(
        "solver_args",
        nargs=argparse.REMAINDER,
        help="Extra solvers.run_solutions arguments after --.",
    )
    args = parser.parse_args(argv)

    if args.solver_args and args.solver_args[0] == "--":
        args.solver_args = args.solver_args[1:]
    for token in args.solver_args:
        option = token.split("=", maxsplit=1)[0]
        if option in FORBIDDEN_SOLVER_OPTIONS:
            parser.error(f"{option} is managed by the collector and cannot be passed after --")

    if args.target_accepted <= 0:
        parser.error("--target-accepted must be positive")
    if args.n_concurrent <= 0:
        parser.error("--n-concurrent must be positive")
    if args.batch_size is None:
        args.batch_size = args.n_concurrent * 4
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must not be negative")
    if args.max_stalled_batches <= 0:
        parser.error("--max-stalled-batches must be positive")
    if not math.isfinite(args.reward_threshold):
        parser.error("--reward-threshold must be finite")
    if not args.model:
        parser.error("--model or MODEL_NAME is required")
    if not args.api_base:
        parser.error("--api-base or OPENAI_API_BASE is required")
    args.api_base = args.api_base.rstrip("/")
    args.api_key = args.api_key or "EMPTY"
    return args


def task_is_complete(task_dir: Path) -> bool:
    return task_dir.is_dir() and all((task_dir / relative).is_file() for relative in REQUIRED_TASK_FILES)


def discover_task_names(tasks_dir: Path, shuffle_seed: int | None = None) -> list[str]:
    names = [path.name for path in tasks_dir.glob("task_*") if task_is_complete(path)]
    if shuffle_seed is None:
        return sorted(names)

    def shuffle_key(name: str) -> bytes:
        return hashlib.sha256(f"{shuffle_seed}:{name}".encode()).digest()

    return sorted(names, key=shuffle_key)


def config_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tasks_dir": str(args.tasks_dir.resolve()),
        "agent": args.agent,
        "model": args.model,
        "api_base": args.api_base,
        "environment": args.env,
        "reward_threshold": args.reward_threshold,
        "shuffle_seed": args.shuffle_seed,
        "docker_network_strategy": args.docker_network_strategy,
        "solver_args": list(args.solver_args),
    }


def load_or_create_state(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    state_path = output / "collector_state.json"
    expected_config = config_snapshot(args)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise SystemExit(f"Unsupported collector state schema in {state_path}")
        actual_config = state.get("config")
        if actual_config != expected_config:
            raise SystemExit(
                "Collector configuration differs from the saved state. "
                f"Saved={actual_config!r}, requested={expected_config!r}"
            )
        saved_target = state.get("target_accepted")
        if saved_target != args.target_accepted:
            raise SystemExit(f"Saved target is {saved_target}; resume requires --target-accepted {saved_target}")
        state["status"] = "running"
        state["consecutive_stalled_batches"] = 0
        state["updated_at"] = timestamp()
        return state

    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty and has no collector state: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "batches").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    now = timestamp()
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "running",
        "config": expected_config,
        "target_accepted": args.target_accepted,
        "attempted_tasks": [],
        "batches": [],
        "consecutive_stalled_batches": 0,
        "created_at": now,
        "updated_at": now,
    }
    atomic_write_json(state_path, state)
    return state


def row_is_usable(row: dict[str, Any]) -> bool:
    trajectory_path = row.get("trajectory_path")
    return bool(row.get("accepted") and trajectory_path and Path(trajectory_path).is_file())


def finalized_row(row: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["collector_batch_index"] = batch["index"]
    result["collector_job_name"] = batch["job_name"]
    if result.get("accepted") and not row_is_usable(result):
        result["accepted"] = False
        result["rejection_reason"] = "collector_missing_trajectory"
    return result


def synthetic_missing_result(task_name: str, batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "trial_name": None,
        "reward": None,
        "accepted": False,
        "accepted_with_timeout": False,
        "original_exception_type": None,
        "rejection_reason": "collector_missing_result",
        "exception_type": None,
        "exception_info": None,
        "result_path": None,
        "trajectory_path": None,
        "materialized_solution_path": None,
        "collector_batch_index": batch["index"],
        "collector_job_name": batch["job_name"],
    }


def rows_for_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    summary_path_value = batch.get("summary_path")
    rows: list[dict[str, Any]] = []
    if summary_path_value:
        summary_path = Path(summary_path_value)
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            raw_rows = summary.get("trials")
            if isinstance(raw_rows, list):
                rows.extend(finalized_row(row, batch) for row in raw_rows if isinstance(row, dict))

    result_task_names = {row.get("task_name") for row in rows if isinstance(row.get("task_name"), str)}
    rows.extend(
        synthetic_missing_result(task_name, batch)
        for task_name in batch.get("scheduled_tasks", [])
        if task_name not in result_task_names
    )
    return rows


def aggregate_rows(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, dict[str, Any]] = {}
    for batch in sorted(state.get("batches", []), key=lambda item: item["index"]):
        for row in rows_for_batch(batch):
            task_name = row.get("task_name")
            if not isinstance(task_name, str):
                continue
            previous = by_task.get(task_name)
            if previous is None or (row_is_usable(row) and not row_is_usable(previous)):
                by_task[task_name] = row

    accepted = sorted(
        (row for row in by_task.values() if row_is_usable(row)),
        key=lambda row: row["task_name"],
    )
    failed = sorted(
        (row for row in by_task.values() if not row_is_usable(row)),
        key=lambda row: row["task_name"],
    )
    return accepted, failed


def write_collection_outputs(
    args: argparse.Namespace,
    state: dict[str, Any],
    discovered_tasks: int,
) -> dict[str, Any]:
    output = args.output.resolve()
    accepted, failed = aggregate_rows(state)
    accepted_path = output / "accepted_trajectories.jsonl"
    failed_path = output / "failed_trials.jsonl"
    atomic_write_jsonl(accepted_path, accepted)
    atomic_write_jsonl(failed_path, failed)

    attempted_count = len(set(state.get("attempted_tasks", [])))
    summary = {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": state["status"],
        "target_accepted": args.target_accepted,
        "accepted_trajectories": len(accepted),
        "remaining_accepted": max(0, args.target_accepted - len(accepted)),
        "failed_tasks": len(failed),
        "attempted_tasks": attempted_count,
        "discovered_complete_tasks": discovered_tasks,
        "available_unattempted_tasks": max(0, discovered_tasks - attempted_count),
        "acceptance_rate": round(len(accepted) / attempted_count, 6) if attempted_count else None,
        "batch_count": len(state.get("batches", [])),
        "consecutive_stalled_batches": state.get("consecutive_stalled_batches", 0),
        "config": state["config"],
        "state_path": str(output / "collector_state.json"),
        "accepted_trajectories_path": str(accepted_path),
        "failed_trials_path": str(failed_path),
        "updated_at": timestamp(),
    }
    atomic_write_json(output / "collection_summary.json", summary)
    state["updated_at"] = summary["updated_at"]
    atomic_write_json(output / "collector_state.json", state)
    return summary


def build_batch_command(
    args: argparse.Namespace,
    selected_tasks: Sequence[str],
    batches_dir: Path,
) -> list[str]:
    command = [
        str(SOLUTION_WRAPPER),
        str(args.tasks_dir.resolve()),
        str(batches_dir),
        "--reward-threshold",
        str(args.reward_threshold),
    ]
    for task_name in selected_tasks:
        command.extend(["--include-task-name", task_name])
    command.extend(args.solver_args)
    return command


def build_batch_env(args: argparse.Namespace, job_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "AGENT": args.agent,
            "MODEL_NAME": args.model,
            "OPENAI_API_BASE": args.api_base,
            "OPENAI_API_KEY": args.api_key,
            "HARBOR_ENV": args.env,
            "JOB_NAME": job_name,
            "N_ATTEMPTS": "1",
            "N_CONCURRENT": str(args.n_concurrent),
            "MAX_RETRIES": str(args.max_retries),
            "PYTHON_BIN": sys.executable,
            "DOCKER_NETWORK_STRATEGY": args.docker_network_strategy,
            "CLEANUP_DOCKER": "1" if args.cleanup_docker else "0",
        }
    )
    return env


def run_batch(command: Sequence[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            message = f"Could not start solution batch: {exc}\n"
            print(message, end="", file=sys.stderr)
            log.write(message)
            return 127

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def finalize_batch(
    batch: dict[str, Any],
    reward_threshold: float,
    returncode: int | None,
) -> None:
    job_dir = Path(batch["job_dir"])
    summary: dict[str, Any] = {"trials": [], "total_trials": 0}
    if job_dir.is_dir():
        summary = summarize_job(job_dir, reward_threshold)
        batch["summary_path"] = summary["summary_path"]

    rows = [row for row in summary.get("trials", []) if isinstance(row, dict)]
    numeric_reward_trials = sum(
        1
        for row in rows
        if isinstance(row.get("reward"), (int, float))
        and not isinstance(row.get("reward"), bool)
        and math.isfinite(float(row["reward"]))
    )
    usable_accepted = sum(row_is_usable(row) for row in rows)
    result_task_names = {row.get("task_name") for row in rows if isinstance(row.get("task_name"), str)}
    scheduled_tasks = batch.get("scheduled_tasks", [])
    batch.update(
        {
            "returncode": returncode,
            "status": "completed" if returncode == 0 else "failed",
            "total_trials": len(rows),
            "usable_accepted_trials": usable_accepted,
            "numeric_reward_trials": numeric_reward_trials,
            "missing_result_tasks": len(set(scheduled_tasks) - result_task_names),
            "finished_at": timestamp(),
        }
    )


def reconcile_interrupted_batches(state: dict[str, Any], reward_threshold: float) -> None:
    for batch in state.get("batches", []):
        if batch.get("status") in {"running", "interrupted"}:
            finalize_batch(batch, reward_threshold, returncode=None)
            batch["status"] = "recovered_after_interruption"


BatchRunner = Callable[[Sequence[str], dict[str, str], Path], int]


def run_collector(
    args: argparse.Namespace,
    batch_runner: BatchRunner | None = None,
) -> int:
    batch_runner = batch_runner or run_batch
    args.tasks_dir = args.tasks_dir.resolve()
    args.output = args.output.resolve()
    if not args.tasks_dir.is_dir():
        raise SystemExit(f"Task directory does not exist: {args.tasks_dir}")
    if not SOLUTION_WRAPPER.is_file():
        raise SystemExit(f"Solution wrapper does not exist: {SOLUTION_WRAPPER}")

    state = load_or_create_state(args)
    reconcile_interrupted_batches(state, args.reward_threshold)
    discovered = discover_task_names(args.tasks_dir, args.shuffle_seed)
    summary = write_collection_outputs(args, state, len(discovered))
    if summary["accepted_trajectories"] >= args.target_accepted:
        state["status"] = "complete"
        write_collection_outputs(args, state, len(discovered))
        return 0

    batches_dir = args.output / "batches"
    logs_dir = args.output / "logs"
    attempted = set(state.get("attempted_tasks", []))

    while True:
        discovered = discover_task_names(args.tasks_dir, args.shuffle_seed)
        available = [task_name for task_name in discovered if task_name not in attempted]
        accepted, _failed = aggregate_rows(state)
        remaining = args.target_accepted - len(accepted)
        if remaining <= 0:
            state["status"] = "complete"
            final_summary = write_collection_outputs(args, state, len(discovered))
            print(json.dumps(final_summary, ensure_ascii=False, indent=2))
            return 0
        if not available:
            state["status"] = "exhausted"
            final_summary = write_collection_outputs(args, state, len(discovered))
            print(json.dumps(final_summary, ensure_ascii=False, indent=2))
            return 2

        selected = available[: min(args.batch_size, remaining)]
        batch_index = len(state["batches"]) + 1
        job_name = f"batch-{batch_index:06d}"
        batch = {
            "index": batch_index,
            "job_name": job_name,
            "job_dir": str(batches_dir / job_name),
            "log_path": str(logs_dir / f"{job_name}.log"),
            "status": "running",
            "scheduled_tasks": selected,
            "started_at": timestamp(),
        }
        state["batches"].append(batch)
        attempted.update(selected)
        state["attempted_tasks"] = sorted(attempted)
        state["status"] = "running"
        write_collection_outputs(args, state, len(discovered))

        command = build_batch_command(args, selected, batches_dir)
        env = build_batch_env(args, job_name)
        print(f"Batch {batch_index}: scheduled={len(selected)} accepted={len(accepted)}/{args.target_accepted}")
        try:
            returncode = batch_runner(command, env, Path(batch["log_path"]))
        except KeyboardInterrupt:
            batch["status"] = "interrupted"
            batch["finished_at"] = timestamp()
            state["status"] = "interrupted"
            write_collection_outputs(args, state, len(discovered))
            raise

        finalize_batch(batch, args.reward_threshold, returncode)
        if batch["numeric_reward_trials"] > 0:
            state["consecutive_stalled_batches"] = 0
        else:
            state["consecutive_stalled_batches"] += 1

        accepted, failed = aggregate_rows(state)
        print(
            f"Batch {batch_index} finished: returncode={returncode} "
            f"accepted={len(accepted)}/{args.target_accepted} "
            f"failed={len(failed)} numeric_rewards={batch['numeric_reward_trials']}"
        )
        write_collection_outputs(args, state, len(discovered))

        if state["consecutive_stalled_batches"] >= args.max_stalled_batches:
            state["status"] = "stalled"
            final_summary = write_collection_outputs(args, state, len(discovered))
            print(json.dumps(final_summary, ensure_ascii=False, indent=2))
            return 3


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_collector(args)
    except KeyboardInterrupt:
        print("Solution collection interrupted; rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
