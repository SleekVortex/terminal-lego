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
from typing import Any, Callable


from generator.contracts import SOQuestion
from generator.instruction_style import (
    INSTRUCTION_REWRITE_MODES,
    validate_lossy_ratio,
)
from generator.llm_client import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    _config,
    call_llm_api,
    configure_trace_writer,
    get_trace_writer,
    token_tracker,
)
from generator.task_builder import TaskGenerator as _TaskBuilder
from generator.resume import build_task_args as _build_task_args
from generator.settings import (
    DEFAULT_INSTRUCTION_REWRITE_MODE,
    DEFAULT_LOSSY_INSTRUCTION_RATIO,
)


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

    def reset(self) -> None:
        self.value = 0


progress_counter = Counter()
error_counter = Counter()
skip_counter = Counter()
_instruction_rewrite_mode = DEFAULT_INSTRUCTION_REWRITE_MODE
_lossy_instruction_ratio = DEFAULT_LOSSY_INSTRUCTION_RATIO


class TaskGenerator(_TaskBuilder):
    """Bind the orchestrator to the CLI-configured LLM client."""

    def __init__(self, question: SOQuestion, output_dir: Path, index: int | None = None) -> None:
        super().__init__(
            question,
            output_dir,
            index=index,
            llm_call=lambda *args, **kwargs: call_llm_api(*args, **kwargs),
            instruction_rewrite_mode=_instruction_rewrite_mode,
            lossy_instruction_ratio=_lossy_instruction_ratio,
        )


def process_question(args: tuple) -> bool:
    question_data, output_dir, total_count, index = args
    generator = None
    trace_started = False
    success = False
    error = None
    try:
        question = SOQuestion.from_dict(question_data)
        generator = TaskGenerator(question, output_dir, index=index)
        trace_writer = get_trace_writer()
        if trace_writer is not None:
            trace_writer.task_started(generator.task_name, question)
            trace_started = True
        success = generator.generate()
        current = progress_counter.increment()
        if success:
            logger.info("Progress: %s/%s - OK: %s", current, total_count, generator.task_name)
        else:
            error_counter.increment()
            logger.error("Progress: %s/%s - FAIL: %s", current, total_count, generator.task_name)
        return success
    except Exception as exc:
        error = str(exc)
        error_counter.increment()
        logger.exception("Exception processing question: %s", exc)
        return False
    finally:
        if trace_started and generator is not None:
            try:
                trace_writer = get_trace_writer()
                if trace_writer is not None:
                    trace_writer.task_finished(
                        generator.task_name,
                        success,
                        error or ("generation_failed" if not success else None),
                    )
            except Exception as exc:
                logger.warning("Failed to finish trace for %s: %s", generator.task_name, exc)


def configure_generation(
    *,
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    instruction_rewrite_mode: str,
    lossy_instruction_ratio: float,
) -> None:
    """Configure the existing generator runtime without changing its logic."""
    global _instruction_rewrite_mode, _lossy_instruction_ratio

    _lossy_instruction_ratio = validate_lossy_ratio(lossy_instruction_ratio)
    _instruction_rewrite_mode = instruction_rewrite_mode
    _config["api_base"] = api_base or os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)
    _config["api_key"] = api_key or os.environ.get("OPENAI_API_KEY", "")
    _config["model"] = model or os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    if not _config["api_key"]:
        raise ValueError("No API key provided. Use --api-key or set OPENAI_API_KEY env var.")


def run_generation(
    *,
    input_path: Path,
    output_dir: Path,
    workers: int,
    start: int = 0,
    limit: int | None = None,
    resume: bool = False,
    trace_dir: Path | None = None,
    on_generated: Callable[[Path, int], None] | None = None,
) -> dict[str, Any]:
    """Generate tasks and notify the caller as soon as each task is complete."""
    for counter in (progress_counter, error_counter, skip_counter):
        counter.reset()

    logger.info("Model: %s", _config["model"])
    logger.info("API base: %s", _config["api_base"])
    logger.info(
        "Instruction rewrite: mode=%s lossy_ratio=%.3f",
        _instruction_rewrite_mode,
        _lossy_instruction_ratio,
    )

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    questions = [q for q in data["questions"] if q.get("accepted_answer")]
    logger.info(
        "Found %s questions with accepted answers (of %s total)",
        len(questions),
        len(data["questions"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_writer = configure_trace_writer(str(trace_dir.resolve()) if trace_dir else None)
    if trace_writer is not None:
        logger.info("Generation traces: %s", trace_writer.output_dir)

    task_args, resume_info = _build_task_args(
        questions,
        output_dir,
        start=start,
        limit=limit,
        resume=resume,
    )
    total_count = len(task_args)

    if resume:
        logger.info(
            "Resume enabled: skipped %s existing question ids, next task index is %s",
            resume_info["skipped_existing_questions"],
            resume_info["next_task_index"],
        )

    logger.info("Processing %s questions (start=%s, resume=%s)", total_count, start, resume)
    start_time = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_question, task_arg): task_arg
            for task_arg in task_args
        }
        for future in as_completed(futures):
            try:
                success = bool(future.result())
            except Exception as exc:
                logger.exception("Future exception: %s", exc)
                continue

            success_count += int(success)
            if success and on_generated is not None:
                task_arg = futures[future]
                task_index = int(task_arg[3])
                task_dir = Path(task_arg[1]) / f"task_{task_index:05d}"
                on_generated(task_dir, total_count)

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
        "trace_dir": str(trace_writer.output_dir) if trace_writer is not None else None,
        "trace_run_id": trace_writer.run_id if trace_writer is not None else None,
        "instruction_rewrite_mode": _instruction_rewrite_mode,
        "lossy_instruction_ratio": _lossy_instruction_ratio,
    }
    with (output_dir / "generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


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
    parser.add_argument(
        "--instruction-rewrite-mode",
        choices=INSTRUCTION_REWRITE_MODES,
        default=os.environ.get(
            "INSTRUCTION_REWRITE_MODE",
            DEFAULT_INSTRUCTION_REWRITE_MODE,
        ),
        help="Instruction rewrite policy: off, concise, lossy, or deterministic mixed.",
    )
    parser.add_argument(
        "--lossy-instruction-ratio",
        type=float,
        default=float(
            os.environ.get(
                "LOSSY_INSTRUCTION_RATIO",
                DEFAULT_LOSSY_INSTRUCTION_RATIO,
            )
        ),
        help="Lossy share used by mixed mode. Must be between 0 and 1.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Write task-generation request/response events as task-specific JSONL files.",
    )
    args = parser.parse_args()

    try:
        configure_generation(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            instruction_rewrite_mode=args.instruction_rewrite_mode,
            lossy_instruction_ratio=args.lossy_instruction_ratio,
        )
    except ValueError as exc:
        parser.error(str(exc))

    run_generation(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        workers=args.workers,
        start=args.start,
        limit=args.limit,
        resume=args.resume,
        trace_dir=args.trace_dir,
    )


if __name__ == "__main__":
    main()
