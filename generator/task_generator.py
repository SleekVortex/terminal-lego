#!/usr/bin/env python3
"""CLI entrypoint for Terminal-Lego task generation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from generator.contracts import SOQuestion
from generator.llm_client import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    _config,
    call_llm_api,
    token_tracker,
)
from generator.task_builder import TaskGenerator as _TaskBuilder
from generator.resume import build_task_args as _build_task_args


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler("task_generator.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value


progress_counter = Counter()
error_counter = Counter()
skip_counter = Counter()


class TaskGenerator(_TaskBuilder):
    """Bind the orchestrator to the CLI-configured LLM client."""

    def __init__(self, question: SOQuestion, output_dir: Path, index: int | None = None) -> None:
        super().__init__(question, output_dir, index=index, llm_call=lambda *args, **kwargs: call_llm_api(*args, **kwargs))


def process_question(args: tuple) -> bool:
    question_data, output_dir, total_count, index = args
    try:
        question = SOQuestion.from_dict(question_data)
        generator = TaskGenerator(question, output_dir, index=index)
        success = generator.generate()
        current = progress_counter.increment()
        if success:
            logger.info("Progress: %s/%s - OK: %s", current, total_count, generator.task_name)
        else:
            error_counter.increment()
            logger.error("Progress: %s/%s - FAIL: %s", current, total_count, generator.task_name)
        return success
    except Exception as exc:
        error_counter.increment()
        logger.exception("Exception processing question: %s", exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal-Lego task generator")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with questions")
    parser.add_argument("--output", "-o", default="./candidates", help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Parallel worker threads")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit questions to process")
    parser.add_argument("--start", "-s", type=int, default=0, help="Start index")
    parser.add_argument("--resume", action="store_true", help="Skip complete tasks with existing source question ids")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    args = parser.parse_args()

    _config["api_base"] = args.api_base or os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)
    _config["api_key"] = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    _config["model"] = args.model or os.environ.get("MODEL_NAME", DEFAULT_MODEL)

    if not _config["api_key"]:
        logger.error("No API key provided. Use --api-key or set OPENAI_API_KEY env var.")
        raise SystemExit(1)

    logger.info("Model: %s", _config["model"])
    logger.info("API base: %s", _config["api_base"])

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    questions = [q for q in data["questions"] if q.get("accepted_answer")]
    logger.info("Found %s questions with accepted answers (of %s total)", len(questions), len(data["questions"]))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_args, resume_info = _build_task_args(
        questions,
        output_dir,
        start=args.start,
        limit=args.limit,
        resume=args.resume,
    )
    total_count = len(task_args)

    if args.resume:
        logger.info(
            "Resume enabled: skipped %s existing question ids, next task index is %s",
            resume_info["skipped_existing_questions"],
            resume_info["next_task_index"],
        )

    logger.info("Processing %s questions (start=%s, resume=%s)", total_count, args.start, args.resume)
    start_time = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_question, arg) for arg in task_args]
        for future in as_completed(futures):
            try:
                success_count += int(bool(future.result()))
            except Exception as exc:
                logger.exception("Future exception: %s", exc)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Done! Time: %.1fs", elapsed)
    logger.info("Success: %s/%s", success_count, total_count)
    logger.info("Failed: %s, Skipped: %s", error_counter.value, skip_counter.value)
    logger.info("Token usage: %s", token_tracker.get_summary())
    logger.info("=" * 60)

    summary = {
        "total_processed": total_count,
        "success_count": success_count,
        "error_count": error_counter.value,
        "skip_count": skip_counter.value,
        "resume": resume_info,
        "elapsed_seconds": elapsed,
        "model": _config["model"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "token_usage": token_tracker.get_summary(),
    }
    with open(output_dir / "generation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
