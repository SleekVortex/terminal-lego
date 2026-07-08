#!/usr/bin/env python3
"""Run the full Terminal-Lego task generation pipeline.

Pipeline:
1. prepare StackOverflow seed
2. generate task candidates
3. validate baseline candidates
4. collect failed tasks
5. regenerate Dockerfiles for eligible failures
6. validate regenerated tasks
7. diagnose remaining failures
8. repair diagnosed failures with validation
9. merge all accepted tasks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from generator import llm_client
from generator.contracts import FailureDiagnosis, RepairAttempt, SOQuestion
from generator.failure.classifier import classify_failure
from generator.llm_client import DEFAULT_API_BASE, DEFAULT_MODEL, TokenTracker
from generator.repair.loop import RepairLoop
from generator.resume import is_complete_task_dir
from generator.task_builder import TaskGenerator
from scripts.regenerate_dockerfiles import regenerate_one
from scripts.validation_first_pipeline import has_required_regen_artifacts
from validator.validate_tasks import TaskValidator


REPO_ROOT = Path(__file__).resolve().parents[1]
PASSED_STATUS = "passed"


def split_categories(values: Sequence[str]) -> list[str]:
    categories: list[str] = []
    for value in values:
        categories.extend(item.strip() for item in value.split(",") if item.strip())
    return categories


def count_task_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.iterdir() if child.is_dir() and child.name.startswith("task_"))


def task_index(task_dir: Path) -> int:
    try:
        return int(task_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_prepare_filters(args: argparse.Namespace, command: list[str]) -> None:
    command.extend(["--start", str(args.start)])
    if args.min_score is not None:
        command.extend(["--min-score", str(args.min_score)])
    for category in split_categories(args.category or []):
        command.extend(["--category", category])
    if args.per_category is not None:
        command.extend(["--per-category", str(args.per_category)])
    elif args.sample_size is not None:
        command.extend(["--sample-size", str(args.sample_size), "--distribution", args.distribution])
    elif args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.sort_by_score:
        command.append("--sort-by-score")


def run_command(
    name: str,
    command: list[str],
    log_path: Path,
    marker_path: Path,
    resume: bool,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if resume and marker_path.exists():
        marker = load_json(marker_path)
        if marker.get("status") == "completed":
            return {"stage": name, "status": "skipped", "marker": str(marker_path)}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()

    elapsed = round(time.time() - started, 3)
    marker = {
        "stage": name,
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "command": command,
        "log": str(log_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(marker_path, marker)
    if returncode != 0:
        raise SystemExit(f"Stage {name!r} failed with return code {returncode}. See {log_path}")
    return marker


def filter_jsonl(input_path: Path, output_path: Path, key: str, expected_value: Any) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        if input_path.exists():
            for line in input_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get(key) == expected_value:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    return count


def build_summary(args: argparse.Namespace, paths: dict[str, Path], stages: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_report = load_json(paths["baseline_validation"] / "validation_report.json")
    regenerated_report = load_json(paths["regenerated_validation"] / "validation_report.json")
    repair_summary = load_json(paths["repair"] / "repair_summary.json")
    accepted_report = load_json(paths["accepted"] / "final_accepted_report.json")
    return {
        "run_dir": str(args.output),
        "input": str(args.input),
        "seed": str(paths["seed"]),
        "candidates": str(paths["candidates"]),
        "accepted": str(paths["accepted"]),
        "counts": {
            "candidates": count_task_dirs(paths["candidates"]),
            "baseline_accepted": count_task_dirs(paths["baseline_validation"]),
            "regenerated_accepted": count_task_dirs(paths["regenerated_validation"]),
            "repair_accepted": count_task_dirs(paths["repair"] / "repaired_validated"),
            "final_accepted": accepted_report.get("final_accepted", count_task_dirs(paths["accepted"])),
        },
        "reports": {
            "baseline_validation": str(paths["baseline_validation"] / "validation_report.json"),
            "baseline_failures": str(paths["baseline_failures"]),
            "dockerfile_regen": str(paths["dockerfile_regen"] / "regenerate_dockerfiles_summary.json"),
            "regenerated_validation": str(paths["regenerated_validation"] / "validation_report.json"),
            "regenerated_failures": str(paths["regenerated_failures"]),
            "diagnoses": str(paths["diagnoses"]),
            "repair": str(paths["repair"] / "repair_summary.json"),
            "accepted": str(paths["accepted"] / "final_accepted_report.json"),
        },
        "baseline_status_counts": baseline_report.get("status_counts", {}),
        "regenerated_status_counts": regenerated_report.get("status_counts", {}),
        "repair_status_counts": repair_summary.get("status_counts", {}),
        "stages": stages,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "seed": run_dir / "seed.json",
        "candidates": run_dir / "candidates",
        "baseline_validation": run_dir / "validation" / "baseline",
        "baseline_failures": run_dir / "failures" / "baseline_failures.jsonl",
        "dockerfile_regen_tasks": run_dir / "failures" / "dockerfile_regen_tasks.jsonl",
        "dockerfile_regen": run_dir / "dockerfile_regen",
        "regenerated_validation": run_dir / "validation" / "dockerfile_regen",
        "regenerated_failures": run_dir / "failures" / "regenerated_failures.jsonl",
        "diagnoses": run_dir / "diagnoses" / "remaining_failures.jsonl",
        "repair": run_dir / "repair",
        "accepted": run_dir / "accepted",
        "logs": run_dir / "pipeline_logs",
        "state": run_dir / "pipeline_state",
        "task_state": run_dir / "pipeline_state" / "tasks",
    }


class ResourceGate:
    def __init__(self, max_concurrency: int):
        if max_concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.semaphore = threading.BoundedSemaphore(max_concurrency)

    def call(self, func, *args, **kwargs):
        with self.semaphore:
            return func(*args, **kwargs)


def configure_llm(args: argparse.Namespace, run_dir: Path) -> None:
    llm_client.token_tracker = TokenTracker(str(run_dir / "api_token_usage.json"))
    llm_client._config["api_base"] = args.api_base or os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)
    llm_client._config["api_key"] = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    llm_client._config["model"] = args.model or os.environ.get("MODEL_NAME", DEFAULT_MODEL)


def load_seed_questions(seed_path: Path) -> list[dict[str, Any]]:
    data = json.loads(seed_path.read_text(encoding="utf-8"))
    return [question for question in data.get("questions", []) if question.get("accepted_answer")]


def append_jsonl_locked(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_validation_report(path: Path, rows: list[dict[str, Any]], elapsed: float) -> None:
    counts = status_counts(rows)
    report = {
        "total": len(rows),
        "passed": counts.get(PASSED_STATUS, 0),
        "failed": counts.get("failed", 0),
        "build_failed": counts.get("build_failed", 0),
        "errors": counts.get("error", 0),
        "status_counts": counts,
        "elapsed_seconds": round(elapsed, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": rows,
    }
    write_json(path / "validation_report.json", report)


def copy_task(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def make_failure_row(result: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    regen_eligible = has_required_regen_artifacts(task_dir)
    return {
        "task": result.get("task"),
        "baseline_status": result.get("status"),
        "baseline_reward": result.get("reward"),
        "baseline_error": result.get("error"),
        "regen_eligible": regen_eligible,
        "final_status": "needs_dockerfile_regen" if regen_eligible else "discard_missing_artifacts",
    }


class StreamingTaskPipeline:
    def __init__(self, args: argparse.Namespace, paths: dict[str, Path]):
        self.args = args
        self.paths = paths
        self.llm_gate = ResourceGate(args.llm_concurrency)
        self.docker_gate = ResourceGate(args.docker_concurrency)
        self.write_lock = threading.RLock()
        self.progress_lock = threading.Lock()
        self.completed = 0
        self.started_at = time.time()
        self.baseline_results: list[dict[str, Any]] = []
        self.regenerated_results: list[dict[str, Any]] = []
        self.regen_rows: list[dict[str, Any]] = []
        self.repair_attempts: list[dict[str, Any]] = []
        self.final_rows: list[dict[str, Any]] = []

    def llm_call(self, *args, **kwargs):
        return self.llm_gate.call(llm_client.call_llm_api, *args, **kwargs)

    def validate_task(self, task_dir: Path, output_dir: Path, timeout: int) -> dict[str, Any]:
        return self.docker_gate.call(TaskValidator(output_dir, timeout, emit_progress=False).validate, task_dir, 1)

    def task_state_path(self, task_name: str) -> Path:
        return self.paths["task_state"] / f"{task_name}.json"

    def load_completed_state(self, task_name: str) -> dict[str, Any] | None:
        state_path = self.task_state_path(task_name)
        if not self.args.resume or not state_path.exists():
            return None
        state = load_json(state_path)
        if state.get("done"):
            return state
        return None

    def write_task_state(self, task_name: str, state: dict[str, Any]) -> None:
        state["task"] = task_name
        state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(self.task_state_path(task_name), state)

    def record_progress(self, task_name: str, status: str, total: int) -> None:
        with self.progress_lock:
            self.completed += 1
            elapsed = max(time.time() - self.started_at, 1e-6)
            rate = self.completed / elapsed * 3600
            print(
                f"Progress: {self.completed}/{total} task={task_name} "
                f"status={status} rate_per_hour={rate:.1f}",
                flush=True,
            )

    def accept_task(self, task_name: str, source: str, source_dir: Path) -> dict[str, Any]:
        copied = copy_task(source_dir / task_name, self.paths["accepted"] / task_name)
        row = {
            "task": task_name,
            "source": source,
            "copied": copied,
            "duplicate": False,
        }
        with self.write_lock:
            self.final_rows.append(row)
        return row

    def generate_candidate(self, question_data: dict[str, Any], index: int) -> tuple[str, bool]:
        question = SOQuestion.from_dict(question_data)
        generator = TaskGenerator(question, self.paths["candidates"], index=index, llm_call=self.llm_call)
        task_name = generator.task_name
        if self.args.resume and is_complete_task_dir(generator.task_dir):
            return task_name, True
        return task_name, bool(generator.generate())

    def regenerate_dockerfile(self, task_dir: Path) -> dict[str, Any]:
        row = self.llm_gate.call(
            regenerate_one,
            task_dir,
            self.paths["candidates"],
            4000,
            False,
        )
        with self.write_lock:
            self.regen_rows.append(row)
            append_jsonl_locked(self.paths["dockerfile_regen"] / "regenerate_dockerfiles.jsonl", row, self.write_lock)
        return row

    def diagnose(self, task_name: str, validation_dir: Path, validation_result: dict[str, Any]) -> FailureDiagnosis:
        log_dir = validation_result.get("log_dir")
        if log_dir:
            diagnosis = classify_failure(task_name, validation_dir / str(log_dir))
        else:
            diagnosis = FailureDiagnosis(
                task_name=task_name,
                failure_class=str(validation_result.get("status") or "unknown"),
                confidence=0.3,
                evidence=[str(validation_result.get("error") or "")],
                allowed_repairs=[],
            )
        append_jsonl_locked(self.paths["diagnoses"], diagnosis.to_dict(), self.write_lock)
        return diagnosis

    def repair(self, diagnosis: FailureDiagnosis) -> list[RepairAttempt]:
        def validate_one(task_dir: Path, validation_output: Path, timeout: int) -> dict[str, Any]:
            return self.validate_task(task_dir, validation_output, timeout)

        loop = RepairLoop(
            self.paths["candidates"],
            self.paths["repair"],
            max_attempts=self.args.repair_max_attempts,
            validate=True,
            timeout=self.args.timeout,
            llm_call=self.llm_call,
            validate_task=validate_one,
        )
        attempts = loop.run_one(diagnosis)
        for attempt in attempts:
            row = attempt.to_dict()
            with self.write_lock:
                self.repair_attempts.append(row)
                append_jsonl_locked(self.paths["repair"] / "repair_report.jsonl", row, self.write_lock)
        return attempts

    def process_one(self, index: int, question_data: dict[str, Any], total: int) -> dict[str, Any]:
        task_name = f"task_{index:05d}"
        completed_state = self.load_completed_state(task_name)
        if completed_state is not None:
            self.record_progress(task_name, str(completed_state.get("final_status") or "skipped"), total)
            return completed_state

        started = time.time()
        state: dict[str, Any] = {
            "done": False,
            "question_id": question_data.get("question_id"),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            task_name, generated = self.generate_candidate(question_data, index)
            task_dir = self.paths["candidates"] / task_name
            state["generated"] = generated
            if not generated:
                state.update(done=True, final_status="generation_failed", elapsed_seconds=round(time.time() - started, 3))
                self.write_task_state(task_name, state)
                self.record_progress(task_name, "generation_failed", total)
                return state

            baseline = self.validate_task(task_dir, self.paths["baseline_validation"], self.args.timeout)
            with self.write_lock:
                self.baseline_results.append(baseline)
            state["baseline_validation"] = baseline
            if baseline.get("status") == PASSED_STATUS and baseline.get("reward") == 1.0:
                self.accept_task(task_name, "baseline", self.paths["baseline_validation"])
                state.update(done=True, final_status="accepted", accepted_source="baseline")
                state["elapsed_seconds"] = round(time.time() - started, 3)
                self.write_task_state(task_name, state)
                self.record_progress(task_name, "accepted:baseline", total)
                return state

            failure_row = make_failure_row(baseline, task_dir)
            append_jsonl_locked(self.paths["baseline_failures"], failure_row, self.write_lock)
            if failure_row["regen_eligible"]:
                append_jsonl_locked(self.paths["dockerfile_regen_tasks"], {"task": task_name}, self.write_lock)

            last_validation = baseline
            last_validation_dir = self.paths["baseline_validation"]
            if (
                not self.args.skip_dockerfile_regeneration
                and failure_row["regen_eligible"]
            ):
                regen = self.regenerate_dockerfile(task_dir)
                state["dockerfile_regen"] = regen
                regenerated = self.validate_task(task_dir, self.paths["regenerated_validation"], self.args.timeout)
                with self.write_lock:
                    self.regenerated_results.append(regenerated)
                state["regenerated_validation"] = regenerated
                last_validation = regenerated
                last_validation_dir = self.paths["regenerated_validation"]
                if regenerated.get("status") == PASSED_STATUS and regenerated.get("reward") == 1.0:
                    self.accept_task(task_name, "regenerated", self.paths["regenerated_validation"])
                    state.update(done=True, final_status="accepted", accepted_source="regenerated")
                    state["elapsed_seconds"] = round(time.time() - started, 3)
                    self.write_task_state(task_name, state)
                    self.record_progress(task_name, "accepted:regenerated", total)
                    return state
                append_jsonl_locked(self.paths["regenerated_failures"], make_failure_row(regenerated, task_dir), self.write_lock)

            if not self.args.skip_repair:
                diagnosis = self.diagnose(task_name, last_validation_dir, last_validation)
                state["diagnosis"] = diagnosis.to_dict()
                attempts = self.repair(diagnosis)
                state["repair_attempts"] = [attempt.to_dict() for attempt in attempts]
                accepted = any(attempt.status == "accepted" for attempt in attempts)
                if accepted:
                    self.accept_task(task_name, "repair", self.paths["repair"] / "repaired_validated")
                    state.update(done=True, final_status="accepted", accepted_source="repair")
                    state["elapsed_seconds"] = round(time.time() - started, 3)
                    self.write_task_state(task_name, state)
                    self.record_progress(task_name, "accepted:repair", total)
                    return state

            state.update(done=True, final_status=str(last_validation.get("status") or "failed"))
            state["elapsed_seconds"] = round(time.time() - started, 3)
            self.write_task_state(task_name, state)
            self.record_progress(task_name, str(state["final_status"]), total)
            return state
        except Exception as exc:
            state.update(done=True, final_status="pipeline_error", error=str(exc), elapsed_seconds=round(time.time() - started, 3))
            self.write_task_state(task_name, state)
            self.record_progress(task_name, "pipeline_error", total)
            return state

    def process_existing_task(self, task_dir: Path, total: int) -> dict[str, Any]:
        task_name = task_dir.name
        completed_state = self.load_completed_state(task_name)
        if completed_state is not None:
            self.record_progress(task_name, str(completed_state.get("final_status") or "skipped"), total)
            return completed_state

        started = time.time()
        state: dict[str, Any] = {
            "done": False,
            "generated": True,
            "existing_candidate": True,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            if not is_complete_task_dir(task_dir):
                state.update(done=True, final_status="incomplete_candidate", elapsed_seconds=round(time.time() - started, 3))
                self.write_task_state(task_name, state)
                self.record_progress(task_name, "incomplete_candidate", total)
                return state

            baseline = self.validate_task(task_dir, self.paths["baseline_validation"], self.args.timeout)
            with self.write_lock:
                self.baseline_results.append(baseline)
            state["baseline_validation"] = baseline
            if baseline.get("status") == PASSED_STATUS and baseline.get("reward") == 1.0:
                self.accept_task(task_name, "baseline", self.paths["baseline_validation"])
                state.update(done=True, final_status="accepted", accepted_source="baseline")
                state["elapsed_seconds"] = round(time.time() - started, 3)
                self.write_task_state(task_name, state)
                self.record_progress(task_name, "accepted:baseline", total)
                return state

            failure_row = make_failure_row(baseline, task_dir)
            append_jsonl_locked(self.paths["baseline_failures"], failure_row, self.write_lock)
            if failure_row["regen_eligible"]:
                append_jsonl_locked(self.paths["dockerfile_regen_tasks"], {"task": task_name}, self.write_lock)

            last_validation = baseline
            last_validation_dir = self.paths["baseline_validation"]
            if not self.args.skip_dockerfile_regeneration and failure_row["regen_eligible"]:
                regen = self.regenerate_dockerfile(task_dir)
                state["dockerfile_regen"] = regen
                regenerated = self.validate_task(task_dir, self.paths["regenerated_validation"], self.args.timeout)
                with self.write_lock:
                    self.regenerated_results.append(regenerated)
                state["regenerated_validation"] = regenerated
                last_validation = regenerated
                last_validation_dir = self.paths["regenerated_validation"]
                if regenerated.get("status") == PASSED_STATUS and regenerated.get("reward") == 1.0:
                    self.accept_task(task_name, "regenerated", self.paths["regenerated_validation"])
                    state.update(done=True, final_status="accepted", accepted_source="regenerated")
                    state["elapsed_seconds"] = round(time.time() - started, 3)
                    self.write_task_state(task_name, state)
                    self.record_progress(task_name, "accepted:regenerated", total)
                    return state
                append_jsonl_locked(self.paths["regenerated_failures"], make_failure_row(regenerated, task_dir), self.write_lock)

            if not self.args.skip_repair:
                diagnosis = self.diagnose(task_name, last_validation_dir, last_validation)
                state["diagnosis"] = diagnosis.to_dict()
                attempts = self.repair(diagnosis)
                state["repair_attempts"] = [attempt.to_dict() for attempt in attempts]
                accepted = any(attempt.status == "accepted" for attempt in attempts)
                if accepted:
                    self.accept_task(task_name, "repair", self.paths["repair"] / "repaired_validated")
                    state.update(done=True, final_status="accepted", accepted_source="repair")
                    state["elapsed_seconds"] = round(time.time() - started, 3)
                    self.write_task_state(task_name, state)
                    self.record_progress(task_name, "accepted:repair", total)
                    return state

            state.update(done=True, final_status=str(last_validation.get("status") or "failed"))
            state["elapsed_seconds"] = round(time.time() - started, 3)
            self.write_task_state(task_name, state)
            self.record_progress(task_name, str(state["final_status"]), total)
            return state
        except Exception as exc:
            state.update(done=True, final_status="pipeline_error", error=str(exc), elapsed_seconds=round(time.time() - started, 3))
            self.write_task_state(task_name, state)
            self.record_progress(task_name, "pipeline_error", total)
            return state

    def finalize(self, run_dir: Path, states: list[dict[str, Any]]) -> dict[str, Any]:
        baseline_results = [
            state["baseline_validation"]
            for state in states
            if isinstance(state.get("baseline_validation"), dict)
        ]
        regenerated_results = [
            state["regenerated_validation"]
            for state in states
            if isinstance(state.get("regenerated_validation"), dict)
        ]
        regen_rows = [
            state["dockerfile_regen"]
            for state in states
            if isinstance(state.get("dockerfile_regen"), dict)
        ]
        repair_attempts = [
            attempt
            for state in states
            for attempt in state.get("repair_attempts", [])
            if isinstance(attempt, dict)
        ]

        write_validation_report(self.paths["baseline_validation"], baseline_results, time.time() - self.started_at)
        write_validation_report(self.paths["regenerated_validation"], regenerated_results, time.time() - self.started_at)

        self.paths["baseline_failures"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["dockerfile_regen_tasks"].parent.mkdir(parents=True, exist_ok=True)
        with self.paths["baseline_failures"].open("w", encoding="utf-8") as failures, self.paths["dockerfile_regen_tasks"].open(
            "w",
            encoding="utf-8",
        ) as regen_tasks:
            for result in baseline_results:
                if result.get("status") == PASSED_STATUS:
                    continue
                row = make_failure_row(result, self.paths["candidates"] / str(result.get("task")))
                failures.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row["regen_eligible"]:
                    regen_tasks.write(json.dumps({"task": result.get("task")}, ensure_ascii=False) + "\n")

        self.paths["regenerated_failures"].parent.mkdir(parents=True, exist_ok=True)
        with self.paths["regenerated_failures"].open("w", encoding="utf-8") as failures:
            for result in regenerated_results:
                if result.get("status") == PASSED_STATUS:
                    continue
                row = make_failure_row(result, self.paths["candidates"] / str(result.get("task")))
                failures.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.paths["dockerfile_regen"].mkdir(parents=True, exist_ok=True)
        with (self.paths["dockerfile_regen"] / "regenerate_dockerfiles.jsonl").open("w", encoding="utf-8") as handle:
            for row in regen_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.paths["repair"].mkdir(parents=True, exist_ok=True)
        with (self.paths["repair"] / "repair_report.jsonl").open("w", encoding="utf-8") as handle:
            for row in repair_attempts:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        regen_status_counts: dict[str, int] = {}
        for row in regen_rows:
            status = str(row.get("status") or "unknown")
            regen_status_counts[status] = regen_status_counts.get(status, 0) + 1
        write_json(
            self.paths["dockerfile_regen"] / "regenerate_dockerfiles_summary.json",
            {
                "tasks_dir": str(self.paths["candidates"]),
                "report_dir": str(self.paths["dockerfile_regen"]),
                "model": self.args.model,
                "api_base": self.args.api_base,
                "workers": self.args.llm_concurrency,
                "total": len(regen_rows),
                "changed": sum(1 for row in regen_rows if row.get("changed")),
                "status_counts": regen_status_counts,
                "elapsed_sec": round(time.time() - self.started_at, 3),
            },
        )

        repair_status_counts: dict[str, int] = {}
        for row in repair_attempts:
            status = str(row.get("status") or "unknown")
            repair_status_counts[status] = repair_status_counts.get(status, 0) + 1
        write_json(
            self.paths["repair"] / "repair_summary.json",
            {
                "tasks_dir": str(self.paths["candidates"]),
                "output_dir": str(self.paths["repair"]),
                "total_diagnoses": sum(1 for state in states if state.get("diagnosis")),
                "total_attempts": len(repair_attempts),
                "status_counts": repair_status_counts,
                "report": str(self.paths["repair"] / "repair_report.jsonl"),
            },
        )

        final_rows = []
        for state in states:
            if state.get("final_status") != "accepted":
                continue
            task_name = str(state.get("task"))
            source = str(state.get("accepted_source") or "unknown")
            final_rows.append(
                {
                    "task": task_name,
                    "source": source,
                    "copied": (self.paths["accepted"] / task_name).exists(),
                    "duplicate": False,
                }
            )
        final_rows = sorted(final_rows, key=lambda row: str(row.get("task")))
        write_json(
            self.paths["accepted"] / "final_accepted_report.json",
            {
                "baseline_validated_dir": str(self.paths["baseline_validation"]),
                "regenerated_validated_dir": str(self.paths["regenerated_validation"]),
                "repair_validated_dir": str(self.paths["repair"] / "repaired_validated"),
                "output_dir": str(self.paths["accepted"]),
                "final_accepted": sum(1 for row in final_rows if row.get("copied")),
                "duplicates": 0,
                "rows": final_rows,
            },
        )

        summary = build_summary(
            self.args,
            self.paths,
            [
                {
                    "stage": "streaming_pipeline",
                    "status": "completed",
                    "task_workers": self.args.task_workers,
                    "llm_concurrency": self.args.llm_concurrency,
                    "docker_concurrency": self.args.docker_concurrency,
                    "elapsed_seconds": round(time.time() - self.started_at, 3),
                }
            ],
        )
        summary["mode"] = "streaming"
        summary["state_counts"] = status_counts([
            {"status": state.get("final_status") or "unknown"}
            for state in states
        ])
        summary["token_usage"] = llm_client.token_tracker.get_summary()
        write_json(run_dir / "pipeline_summary.json", summary)
        return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input StackOverflow JSONL dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Pipeline run directory.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="Skip stages with completed markers.")
    parser.add_argument(
        "--mode",
        choices=("streaming", "staged"),
        default="streaming",
        help="streaming runs each task end-to-end; staged preserves the old batch-stage pipeline.",
    )

    parser.add_argument("--generate-workers", type=int, default=1)
    parser.add_argument("--validate-workers", type=int, default=8)
    parser.add_argument("--dockerfile-workers", type=int, default=8)
    parser.add_argument("--repair-workers", type=int, default=1)
    parser.add_argument("--task-workers", type=int, default=None, help="End-to-end task workers for --mode streaming.")
    parser.add_argument("--llm-concurrency", type=int, default=None, help="Max concurrent LLM calls for --mode streaming.")
    parser.add_argument("--docker-concurrency", type=int, default=None, help="Max concurrent Docker validations for --mode streaming.")
    parser.add_argument(
        "--existing-candidates-only",
        action="store_true",
        help="In streaming mode, process existing candidates/task_* dirs directly instead of generating from seed rows.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--repair-max-attempts", type=int, default=2)

    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))

    parser.add_argument("--min-score", type=int, default=None)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--distribution", default="terminal-bench-2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sort-by-score", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--skip-dockerfile-regeneration", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    return parser.parse_args(argv)


def run_staged_pipeline(args: argparse.Namespace) -> int:
    run_dir = args.output.resolve()
    paths = build_paths(run_dir)
    for path in (run_dir, paths["logs"], paths["state"]):
        path.mkdir(parents=True, exist_ok=True)

    py = args.python_bin
    stages: list[dict[str, Any]] = []

    prepare = [py, "-m", "sources.stackoverflow.prepare_dataset", "--input", str(args.input.resolve()), "--output", str(paths["seed"]), "--format", "generator-json"]
    append_prepare_filters(args, prepare)
    stages.append(run_command("prepare_seed", prepare, paths["logs"] / "01_prepare_seed.log", paths["state"] / "01_prepare_seed.json", args.resume))

    generate = [py, "-m", "generator.task_generator", "--input", str(paths["seed"]), "--output", str(paths["candidates"]), "--workers", str(args.generate_workers), "--api-key", args.api_key]
    if args.api_base:
        generate.extend(["--api-base", args.api_base])
    if args.model:
        generate.extend(["--model", args.model])
    if args.resume:
        generate.append("--resume")
    stages.append(run_command("generate_candidates", generate, paths["logs"] / "02_generate_candidates.log", paths["state"] / "02_generate_candidates.json", args.resume))

    baseline_validate = [py, "-m", "validator.validate_tasks", "--input", str(paths["candidates"]), "--output", str(paths["baseline_validation"]), "--workers", str(args.validate_workers), "--timeout", str(args.timeout)]
    stages.append(run_command("validate_baseline", baseline_validate, paths["logs"] / "03_validate_baseline.log", paths["state"] / "03_validate_baseline.json", args.resume))

    collect_baseline = [py, "-m", "scripts.validation_first_pipeline", "collect-failures", "--validation-report", str(paths["baseline_validation"] / "validation_report.json"), "--tasks-dir", str(paths["candidates"]), "--output", str(paths["baseline_failures"])]
    stages.append(run_command("collect_baseline_failures", collect_baseline, paths["logs"] / "04_collect_baseline_failures.log", paths["state"] / "04_collect_baseline_failures.json", args.resume))

    regen_task_count = filter_jsonl(paths["baseline_failures"], paths["dockerfile_regen_tasks"], "regen_eligible", True)
    write_json(paths["state"] / "05_dockerfile_regen_tasks.json", {"stage": "dockerfile_regen_tasks", "count": regen_task_count, "path": str(paths["dockerfile_regen_tasks"])})

    if not args.skip_dockerfile_regeneration and regen_task_count > 0:
        regen = [py, "-m", "scripts.regenerate_dockerfiles", "--tasks-dir", str(paths["candidates"]), "--report-dir", str(paths["dockerfile_regen"]), "--workers", str(args.dockerfile_workers), "--task-list", str(paths["dockerfile_regen_tasks"]), "--api-key", args.api_key]
        if args.api_base:
            regen.extend(["--api-base", args.api_base])
        if args.model:
            regen.extend(["--model", args.model])
        stages.append(run_command("regenerate_dockerfiles", regen, paths["logs"] / "05_regenerate_dockerfiles.log", paths["state"] / "05_regenerate_dockerfiles.json", args.resume))

        validate_regenerated = [py, "-m", "validator.validate_tasks", "--input", str(paths["candidates"]), "--output", str(paths["regenerated_validation"]), "--workers", str(args.validate_workers), "--timeout", str(args.timeout), "--task-list", str(paths["dockerfile_regen_tasks"])]
        stages.append(run_command("validate_regenerated", validate_regenerated, paths["logs"] / "06_validate_regenerated.log", paths["state"] / "06_validate_regenerated.json", args.resume))

        collect_regenerated = [py, "-m", "scripts.validation_first_pipeline", "collect-failures", "--validation-report", str(paths["regenerated_validation"] / "validation_report.json"), "--tasks-dir", str(paths["candidates"]), "--output", str(paths["regenerated_failures"])]
        stages.append(run_command("collect_regenerated_failures", collect_regenerated, paths["logs"] / "07_collect_regenerated_failures.log", paths["state"] / "07_collect_regenerated_failures.json", args.resume))

        diagnose = [py, "-m", "scripts.diagnose_failed_tasks", "--validation-output", str(paths["regenerated_validation"]), "--output", str(paths["diagnoses"])]
        stages.append(run_command("diagnose_remaining_failures", diagnose, paths["logs"] / "08_diagnose_remaining_failures.log", paths["state"] / "08_diagnose_remaining_failures.json", args.resume))
    else:
        paths["regenerated_validation"].mkdir(parents=True, exist_ok=True)
        stages.append({"stage": "regenerate_dockerfiles", "status": "skipped", "reason": "disabled_or_no_eligible_failures"})
        if args.skip_dockerfile_regeneration:
            diagnose = [py, "-m", "scripts.diagnose_failed_tasks", "--validation-output", str(paths["baseline_validation"]), "--output", str(paths["diagnoses"])]
            stages.append(run_command("diagnose_baseline_failures", diagnose, paths["logs"] / "08_diagnose_baseline_failures.log", paths["state"] / "08_diagnose_baseline_failures.json", args.resume))
        else:
            paths["diagnoses"].parent.mkdir(parents=True, exist_ok=True)
            paths["diagnoses"].write_text("", encoding="utf-8")

    diagnosis_count = sum(1 for line in paths["diagnoses"].read_text(encoding="utf-8").splitlines() if line.strip()) if paths["diagnoses"].exists() else 0
    if not args.skip_repair and diagnosis_count > 0:
        repair = [py, "-m", "scripts.repair_failed_tasks", "--tasks-dir", str(paths["candidates"]), "--output", str(paths["repair"]), "--diagnoses", str(paths["diagnoses"]), "--workers", str(args.repair_workers), "--max-attempts", str(args.repair_max_attempts), "--timeout", str(args.timeout), "--api-key", args.api_key]
        if args.api_base:
            repair.extend(["--api-base", args.api_base])
        if args.model:
            repair.extend(["--model", args.model])
        stages.append(run_command("repair_failures", repair, paths["logs"] / "09_repair_failures.log", paths["state"] / "09_repair_failures.json", args.resume))
    else:
        (paths["repair"] / "repaired_validated").mkdir(parents=True, exist_ok=True)
        stages.append({"stage": "repair_failures", "status": "skipped", "reason": "disabled_or_no_diagnoses"})

    merge = [
        py,
        "-m",
        "scripts.validation_first_pipeline",
        "merge-accepted",
        "--baseline-validated-dir",
        str(paths["baseline_validation"]),
        "--regenerated-validated-dir",
        str(paths["regenerated_validation"]),
        "--repair-validated-dir",
        str(paths["repair"] / "repaired_validated"),
        "--output-dir",
        str(paths["accepted"]),
    ]
    stages.append(run_command("merge_accepted", merge, paths["logs"] / "10_merge_accepted.log", paths["state"] / "10_merge_accepted.json", False))

    summary = build_summary(args, paths, stages)
    write_json(run_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def resolve_streaming_concurrency(args: argparse.Namespace) -> None:
    if args.llm_concurrency is None:
        args.llm_concurrency = max(args.generate_workers, args.dockerfile_workers, args.repair_workers, 1)
    if args.docker_concurrency is None:
        args.docker_concurrency = max(args.validate_workers, 1)
    if args.task_workers is None:
        args.task_workers = max(args.llm_concurrency, args.docker_concurrency, 1)


def clear_streaming_reports(paths: dict[str, Path], resume: bool) -> None:
    if resume:
        return
    for path in (
        paths["baseline_failures"],
        paths["dockerfile_regen_tasks"],
        paths["regenerated_failures"],
        paths["diagnoses"],
        paths["dockerfile_regen"] / "regenerate_dockerfiles.jsonl",
        paths["repair"] / "repair_report.jsonl",
    ):
        if path.exists():
            path.unlink()


def run_streaming_pipeline(args: argparse.Namespace) -> int:
    resolve_streaming_concurrency(args)
    run_dir = args.output.resolve()
    paths = build_paths(run_dir)
    for path in (
        run_dir,
        paths["logs"],
        paths["state"],
        paths["task_state"],
        paths["candidates"],
        paths["baseline_validation"],
        paths["regenerated_validation"],
        paths["dockerfile_regen"],
        paths["repair"],
        paths["accepted"],
    ):
        path.mkdir(parents=True, exist_ok=True)
    clear_streaming_reports(paths, args.resume)
    configure_llm(args, run_dir)

    if args.existing_candidates_only:
        all_task_dirs = sorted(
            (
                path
                for path in paths["candidates"].iterdir()
                if path.is_dir() and path.name.startswith("task_")
            ),
            key=task_index,
        )
        task_dirs = [path for path in all_task_dirs if task_index(path) >= args.start]
        if args.limit is not None:
            task_dirs = task_dirs[: args.limit]
        total = len(task_dirs)
    else:
        py = args.python_bin
        prepare = [
            py,
            "-m",
            "sources.stackoverflow.prepare_dataset",
            "--input",
            str(args.input.resolve()),
            "--output",
            str(paths["seed"]),
            "--format",
            "generator-json",
        ]
        append_prepare_filters(args, prepare)
        run_command(
            "prepare_seed",
            prepare,
            paths["logs"] / "01_prepare_seed.log",
            paths["state"] / "01_prepare_seed.json",
            args.resume,
        )
        questions = load_seed_questions(paths["seed"])
        total = len(questions)

    print(
        json.dumps(
            {
                "mode": "streaming",
                "existing_candidates_only": args.existing_candidates_only,
                "questions": total,
                "task_workers": args.task_workers,
                "llm_concurrency": args.llm_concurrency,
                "docker_concurrency": args.docker_concurrency,
                "model": llm_client._config["model"],
                "api_base": llm_client._config["api_base"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    pipeline = StreamingTaskPipeline(args, paths)
    states: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.task_workers) as executor:
        if args.existing_candidates_only:
            futures = {
                executor.submit(pipeline.process_existing_task, task_dir, total): task_dir
                for task_dir in task_dirs
            }
        else:
            futures = {
                executor.submit(pipeline.process_one, index, question, total): index
                for index, question in enumerate(questions)
            }
        for future in as_completed(futures):
            states.append(future.result())

    summary = pipeline.finalize(run_dir, states)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "staged":
        return run_staged_pipeline(args)
    return run_streaming_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
