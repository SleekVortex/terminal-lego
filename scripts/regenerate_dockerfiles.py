#!/usr/bin/env python3
"""Regenerate Dockerfiles for existing Terminal-Lego task candidates."""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from generator import llm_client
from generator.contracts import SOQuestion
from generator.llm_client import DEFAULT_API_BASE, DEFAULT_MODEL, TokenTracker
from generator.orchestrator import TaskGenerator
from generator.prompt_loader import DOCKERFILE_PROMPT_TEMPLATE, SYSTEM_PROMPT
from generator.reviewers.dockerfile_review import (
    ensure_verifier_deps,
    extract_dockerfile_from_response,
    review_dockerfile,
)
from generator.settings import DOCKERFILE_MAX_ATTEMPTS
from generator.task_writer import format_env_file_list


def parse_task_index(task_name: str) -> Optional[int]:
    match = re.fullmatch(r"task_(\d+)", task_name)
    if not match:
        return None
    return int(match.group(1))


def load_task_toml(path: Path) -> Dict[str, Any]:
    if not path.exists() or tomllib is None:
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path, max_bytes: Optional[int] = None) -> str:
    if not path.exists():
        return ""
    if max_bytes is None:
        return path.read_text(encoding="utf-8", errors="replace")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    text = data[:max_bytes].decode("utf-8", errors="replace")
    if len(data) > max_bytes:
        text += "\n...[truncated]"
    return text


def build_env_data(environment_dir: Path, max_file_bytes: int) -> Dict[str, Any]:
    files: Dict[str, str] = {}
    directories = []
    if not environment_dir.exists():
        return {"files": {}, "directories": ["task_file"]}

    for path in sorted(environment_dir.rglob("*")):
        rel = path.relative_to(environment_dir).as_posix()
        if rel == "Dockerfile":
            continue
        if path.is_dir():
            directories.append(rel)
        elif path.is_file():
            files[rel] = read_text(path, max_bytes=max_file_bytes)

    if "task_file" not in directories and not any(name.startswith("task_file/") for name in directories):
        directories.insert(0, "task_file")
    return {"files": files, "directories": directories}


def make_question(task_dir: Path, task_toml: Dict[str, Any]) -> SOQuestion:
    metadata = task_toml.get("metadata", {}) if isinstance(task_toml, dict) else {}
    index = parse_task_index(task_dir.name) or 0
    question_id = int(metadata.get("source_question_id") or index)
    tags = metadata.get("tags") or []
    categories = metadata.get("categories") or []
    if isinstance(tags, str):
        tags = [tags]
    if isinstance(categories, str):
        categories = [categories]

    return SOQuestion(
        question_id=question_id,
        title=task_dir.name,
        body="",
        tags=list(tags),
        score=int(metadata.get("source_score") or 0),
        accepted_answer_id=None,
        accepted_answer_body=None,
        accepted_answer_score=None,
        link=str(metadata.get("source_url") or ""),
        categories=list(categories),
        selected_category=metadata.get("category"),
    )


def generate_dockerfile_strict(
    generator: TaskGenerator,
    instruction: str,
    env_data: Dict[str, Any],
    solution: str,
    test_outputs_py: str,
) -> tuple[str, int, list[str]]:
    prompt = DOCKERFILE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        tags=", ".join(generator.question.tags),
        env_file_list=format_env_file_list(env_data),
        solution=solution,
        test_outputs_py=test_outputs_py,
    )
    feedback = ""
    last_issues = []
    for attempt in range(DOCKERFILE_MAX_ATTEMPTS):
        attempt_prompt = prompt
        if feedback:
            attempt_prompt += (
                "\n\nPrevious Dockerfile attempt was rejected for these issues:\n"
                + feedback
                + "\nRegenerate only the Dockerfile."
            )
        response = llm_client.call_llm_api(
            attempt_prompt,
            SYSTEM_PROMPT,
            temperature=None,
            task_name=generator.task_name,
            stage="dockerfile",
        )
        if not response:
            last_issues = ["LLM returned no Dockerfile response"]
            feedback = "- " + last_issues[0]
            continue
        extracted = extract_dockerfile_from_response(response)
        if not extracted:
            last_issues = ["LLM Dockerfile response did not contain a fenced Dockerfile block"]
            feedback = "- " + last_issues[0]
            continue
        dockerfile = ensure_verifier_deps(extracted)
        review = review_dockerfile(dockerfile, env_data)
        if review.get("pass"):
            return dockerfile, attempt + 1, []
        last_issues = list(review.get("issues", []))
        feedback = "\n".join(f"- {issue}" for issue in last_issues)
    raise RuntimeError("Dockerfile review failed: " + "; ".join(last_issues))


def load_task_filter(task_list: Optional[Path]) -> Optional[set[str]]:
    if task_list is None:
        return None
    tasks = set()
    for line in task_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_name = row.get("task")
        if task_name:
            tasks.add(task_name)
    return tasks


def iter_task_dirs(tasks_dir: Path, start: int, limit: Optional[int], task_filter: Optional[set[str]]) -> Iterable[Path]:
    task_dirs = sorted(path for path in tasks_dir.iterdir() if path.is_dir() and path.name.startswith("task_"))
    if task_filter is not None:
        task_dirs = [path for path in task_dirs if path.name in task_filter]
    task_dirs = task_dirs[start:]
    if limit is not None:
        task_dirs = task_dirs[:limit]
    return task_dirs


def regenerate_one(
    task_dir: Path,
    tasks_dir: Path,
    max_file_bytes: int,
    dry_run: bool,
) -> Dict[str, Any]:
    started = time.time()
    dockerfile_path = task_dir / "environment" / "Dockerfile"
    try:
        instruction = read_text(task_dir / "instruction.md")
        solution = read_text(task_dir / "solution" / "solve.sh")
        test_outputs_py = read_text(task_dir / "tests" / "test_outputs.py")
        env_data = build_env_data(task_dir / "environment", max_file_bytes=max_file_bytes)
        task_toml = load_task_toml(task_dir / "task.toml")

        generator = TaskGenerator(
            make_question(task_dir, task_toml),
            tasks_dir,
            index=parse_task_index(task_dir.name),
            llm_call=llm_client.call_llm_api,
        )
        dockerfile, attempts, issues = generate_dockerfile_strict(generator, instruction, env_data, solution, test_outputs_py)

        old_text = read_text(dockerfile_path)
        changed = dockerfile.strip() != old_text.strip()
        if changed and not dry_run:
            tmp_path = dockerfile_path.with_suffix(".Dockerfile.tmp")
            tmp_path.write_text(dockerfile.rstrip() + "\n", encoding="utf-8")
            tmp_path.replace(dockerfile_path)

        return {
            "task": task_dir.name,
            "status": "regenerated" if changed else "unchanged",
            "changed": changed,
            "kept_old": False,
            "attempts": attempts,
            "issues": issues,
            "duration_sec": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "task": task_dir.name,
            "status": "kept_old",
            "changed": False,
            "kept_old": True,
            "attempts": DOCKERFILE_MAX_ATTEMPTS,
            "issues": [str(exc)],
            "duration_sec": round(time.time() - started, 3),
            "error": str(exc),
        }


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-list", type=Path, default=None, help="JSONL list with a 'task' field; only these tasks are processed.")
    parser.add_argument("--max-file-bytes", type=int, default=4000)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "regenerate_dockerfiles.jsonl"
    summary_path = args.report_dir / "regenerate_dockerfiles_summary.json"
    llm_client.token_tracker = TokenTracker(str(args.report_dir / "dockerfile_regen_token_usage.json"))
    llm_client._config["api_base"] = args.api_base
    llm_client._config["api_key"] = args.api_key
    llm_client._config["model"] = args.model

    task_filter = load_task_filter(args.task_list)
    task_dirs = list(iter_task_dirs(args.tasks_dir, args.start, args.limit, task_filter))
    total = len(task_dirs)
    print(f"tasks_dir={args.tasks_dir}")
    print(f"report_dir={args.report_dir}")
    print(f"model={args.model}")
    print(f"api_base={args.api_base}")
    print(f"workers={args.workers}")
    print(f"total={total}")
    print(f"dry_run={args.dry_run}")
    if args.task_list:
        print(f"task_list={args.task_list}")

    started = time.time()
    status_counts: Dict[str, int] = {}
    changed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(regenerate_one, task_dir, args.tasks_dir, args.max_file_bytes, args.dry_run): task_dir
            for task_dir in task_dirs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(report_path, row)
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
            changed_count += int(bool(row.get("changed")))
            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - started
                rate = completed / elapsed * 3600 if elapsed > 0 else 0.0
                print(
                    f"Progress: {completed}/{total} changed={changed_count} "
                    f"kept_old={status_counts.get('kept_old', 0)} rate_per_hour={rate:.1f}",
                    flush=True,
                )

    summary = {
        "tasks_dir": str(args.tasks_dir),
        "report_dir": str(args.report_dir),
        "model": args.model,
        "api_base": args.api_base,
        "workers": args.workers,
        "total": total,
        "changed": changed_count,
        "status_counts": status_counts,
        "task_list": str(args.task_list) if args.task_list else None,
        "dry_run": args.dry_run,
        "elapsed_sec": round(time.time() - started, 3),
        "token_usage": llm_client.token_tracker.get_summary(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
