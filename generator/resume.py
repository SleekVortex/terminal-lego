from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


STACKOVERFLOW_QUESTION_URL_RE = re.compile(r"stackoverflow\.com/(?:questions|q)/(\d+)")
TASK_DIR_RE = re.compile(r"^task_(\d+)$")
REQUIRED_COMPLETE_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
    "tests/test_outputs.py",
)


def extract_question_id_from_source_url(source_url: str) -> Optional[int]:
    match = STACKOVERFLOW_QUESTION_URL_RE.search(source_url or "")
    if not match:
        return None
    return int(match.group(1))


def extract_source_question_id_from_task_toml(task_toml: Path) -> Optional[int]:
    try:
        text = task_toml.read_text(encoding="utf-8")
    except OSError:
        return None

    if tomllib is not None:
        try:
            metadata = tomllib.loads(text).get("metadata", {})
            source_question_id = metadata.get("source_question_id")
            if source_question_id is not None:
                return int(source_question_id)
            return extract_question_id_from_source_url(str(metadata.get("source_url") or ""))
        except Exception:
            pass

    id_match = re.search(r"(?m)^source_question_id\s*=\s*\"?(\d+)\"?\s*$", text)
    if id_match:
        return int(id_match.group(1))
    url_match = re.search(r'(?m)^source_url\s*=\s*"([^"]+)"\s*$', text)
    if url_match:
        return extract_question_id_from_source_url(url_match.group(1))
    return None


def is_complete_task_dir(task_dir: Path) -> bool:
    return all((task_dir / relative_path).is_file() for relative_path in REQUIRED_COMPLETE_TASK_FILES)


def collect_existing_question_ids(output_dir: Path) -> dict[int, Path]:
    question_ids: dict[int, Path] = {}
    if not output_dir.exists():
        return question_ids
    for task_dir in sorted(output_dir.glob("task_*")):
        if not task_dir.is_dir() or not is_complete_task_dir(task_dir):
            continue
        question_id = extract_source_question_id_from_task_toml(task_dir / "task.toml")
        if question_id is not None:
            question_ids.setdefault(question_id, task_dir)
    return question_ids


def find_next_task_index(output_dir: Path) -> int:
    max_index = -1
    if output_dir.exists():
        for task_dir in output_dir.glob("task_*"):
            if not task_dir.is_dir():
                continue
            match = TASK_DIR_RE.match(task_dir.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def build_task_args(
    questions: list[dict],
    output_dir: Path,
    start: int = 0,
    limit: Optional[int] = None,
    resume: bool = False,
) -> tuple[list[tuple], dict]:
    indexed_questions = list(enumerate(questions))[start:]
    if limit:
        indexed_questions = indexed_questions[:limit]

    resume_info = {
        "enabled": resume,
        "existing_complete_tasks": 0,
        "existing_question_ids": 0,
        "skipped_existing_questions": 0,
        "next_task_index": None,
    }

    if resume:
        existing_question_ids = collect_existing_question_ids(output_dir)
        next_task_index = find_next_task_index(output_dir)
        resume_info.update({
            "existing_complete_tasks": len(existing_question_ids),
            "existing_question_ids": len(existing_question_ids),
            "next_task_index": next_task_index,
        })
        pending_questions = []
        skipped_existing = 0
        for _original_index, question in indexed_questions:
            try:
                question_id = int(question.get("question_id"))
            except (TypeError, ValueError):
                question_id = None
            if question_id is not None and question_id in existing_question_ids:
                skipped_existing += 1
                continue
            pending_questions.append(question)
        resume_info["skipped_existing_questions"] = skipped_existing
        total_count = len(pending_questions)
        return [
            (question, output_dir, total_count, next_task_index + offset)
            for offset, question in enumerate(pending_questions)
        ], resume_info

    total_count = len(indexed_questions)
    return [
        (question, output_dir, total_count, original_index)
        for original_index, question in indexed_questions
    ], resume_info
