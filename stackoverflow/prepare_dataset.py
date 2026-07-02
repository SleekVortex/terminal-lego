#!/usr/bin/env python3
"""Prepare StackOverflow dump-derived JSONL rows for Terminal-Lego generation.

The input is a JSONL file with rows shaped like the Posts.xml extraction output:
question fields, tags, score, and an accepted_answer object. The output can be
either another classified JSONL file or the generator contract:

    {"metadata": {...}, "questions": [...]}
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from .tag_taxonomy import (
        CATEGORY_ORDER,
        TERMINAL_BENCH_2_DISTRIBUTION,
        classify_tags,
        scaled_quotas,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from tag_taxonomy import (  # type: ignore
        CATEGORY_ORDER,
        TERMINAL_BENCH_2_DISTRIBUTION,
        classify_tags,
        scaled_quotas,
    )


ScoreKey = Tuple[int, int, int, int]


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        if value.startswith("<") and value.endswith(">"):
            return [part for part in value.strip("<>").split("><") if part]
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def normalize_question(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one dump/API row to the shape expected by task_generator.py."""
    question_id = _as_int(row.get("question_id", row.get("Id")))
    if question_id <= 0:
        raise ValueError("row is missing a positive question_id")

    accepted_answer = row.get("accepted_answer") or {}
    if not accepted_answer and row.get("accepted_answer_body"):
        accepted_answer = {
            "answer_id": _as_int(row.get("accepted_answer_id")),
            "body": row.get("accepted_answer_body", ""),
            "score": _as_int(row.get("accepted_answer_score")),
        }

    tags = _string_list(row.get("tags", row.get("Tags")))
    categories = _string_list(row.get("categories"))
    if not categories:
        categories = classify_tags(tags)

    selected_category = row.get("selected_category") or row.get("category")
    if selected_category and selected_category not in categories:
        categories = [selected_category] + categories
    if not selected_category and categories:
        selected_category = categories[0]

    normalized = dict(row)
    normalized.update(
        {
            "question_id": question_id,
            "title": str(row.get("title", row.get("Title", ""))),
            "body": str(
                row.get(
                    "body",
                    row.get("Body", row.get("body_text", "")),
                )
            ),
            "tags": tags,
            "score": _as_int(row.get("score", row.get("Score"))),
            "view_count": _as_int(row.get("view_count", row.get("ViewCount"))),
            "answer_count": _as_int(row.get("answer_count", row.get("AnswerCount"))),
            "accepted_answer_id": _as_int(
                row.get("accepted_answer_id", row.get("AcceptedAnswerId"))
            ),
            "accepted_answer": accepted_answer,
            "link": row.get("link") or f"https://stackoverflow.com/questions/{question_id}",
            "categories": categories,
            "selected_category": selected_category,
        }
    )
    return normalized


def score_key(row: Dict[str, Any]) -> ScoreKey:
    accepted = row.get("accepted_answer") or {}
    return (
        _as_int(row.get("score")),
        _as_int(accepted.get("score")),
        _as_int(row.get("view_count")),
        _as_int(row.get("answer_count")),
    )


def sort_key_desc(row: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    key = score_key(row)
    return (-key[0], -key[1], -key[2], -key[3], _as_int(row.get("question_id")))


def _matches_categories(row: Dict[str, Any], categories: Sequence[str]) -> bool:
    if not categories:
        return True
    row_categories = set(_string_list(row.get("categories")))
    return bool(row_categories & set(categories))


def _category_for_selection(row: Dict[str, Any], allowed: Sequence[str]) -> Optional[str]:
    selected = row.get("selected_category")
    if selected and (not allowed or selected in allowed):
        return str(selected)
    for category in _string_list(row.get("categories")):
        if not allowed or category in allowed:
            return category
    return None


def iter_normalized_filtered(
    input_path: Path,
    min_score: Optional[int],
    categories: Sequence[str],
    allow_uncategorized: bool,
) -> Iterator[Dict[str, Any]]:
    for row in iter_jsonl(input_path):
        normalized = normalize_question(row)
        if not normalized.get("accepted_answer"):
            continue
        if min_score is not None and _as_int(normalized.get("score")) < min_score:
            continue
        if not normalized.get("categories") and not allow_uncategorized:
            continue
        if not _matches_categories(normalized, categories):
            continue
        yield normalized


def select_first(
    rows: Iterable[Dict[str, Any]],
    start: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index < start:
            continue
        if limit is not None and len(selected) >= limit:
            break
        selected.append(row)
    return selected


def select_top(
    rows: Iterable[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    heap: List[Tuple[ScoreKey, int, Dict[str, Any]]] = []
    for seq, row in enumerate(rows):
        item = (score_key(row), seq, row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda item: item[2] and sort_key_desc(item[2]))]


def select_per_category(
    rows: Iterable[Dict[str, Any]],
    per_category: int,
    categories: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = list(categories) if categories else CATEGORY_ORDER
    heaps: Dict[str, List[Tuple[ScoreKey, int, Dict[str, Any]]]] = {
        category: [] for category in allowed
    }
    for seq, row in enumerate(rows):
        category = _category_for_selection(row, allowed)
        if not category:
            continue
        row["selected_category"] = category
        heap = heaps[category]
        item = (score_key(row), seq, row)
        if len(heap) < per_category:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    selected: List[Dict[str, Any]] = []
    for category in allowed:
        selected.extend(item[2] for item in sorted(heaps[category], key=lambda item: sort_key_desc(item[2])))
    return selected


def select_benchmark_distribution(
    rows: Iterable[Dict[str, Any]],
    sample_size: int,
    categories: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = list(categories) if categories else CATEGORY_ORDER
    base_distribution = {
        category: TERMINAL_BENCH_2_DISTRIBUTION[category]
        for category in allowed
        if category in TERMINAL_BENCH_2_DISTRIBUTION
    }
    quotas = scaled_quotas(sample_size, base_distribution)
    heaps: Dict[str, List[Tuple[ScoreKey, int, Dict[str, Any]]]] = {
        category: [] for category in quotas
    }
    counts = {category: 0 for category in quotas}

    for seq, row in enumerate(rows):
        category = _category_for_selection(row, allowed)
        if category not in quotas:
            continue
        row = dict(row)
        row["selected_category"] = category
        counts[category] += 1

        # Keep enough overflow to fill deficits from categories with low supply.
        cap = max(quotas[category] * 3, quotas[category] + 5000, 2000)
        heap = heaps[category]
        item = (score_key(row), seq, row)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    targets = {category: min(quotas[category], counts[category]) for category in quotas}
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    leftovers: List[Dict[str, Any]] = []

    for category in quotas:
        candidates = [item[2] for item in sorted(heaps[category], key=lambda item: sort_key_desc(item[2]))]
        for row in candidates[: targets[category]]:
            qid = row["question_id"]
            if qid not in selected_ids:
                selected.append(row)
                selected_ids.add(qid)
        leftovers.extend(candidates[targets[category] :])

    if len(selected) < sample_size:
        for row in sorted(leftovers, key=sort_key_desc):
            qid = row["question_id"]
            if qid in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(qid)
            if len(selected) >= sample_size:
                break

    return selected[:sample_size]


def write_generator_json(rows: Sequence[Dict[str, Any]], output_path: Path, metadata: Dict[str, Any]) -> None:
    payload = {
        "metadata": {
            **metadata,
            "total": len(rows),
            "with_answers": sum(1 for row in rows if row.get("accepted_answer")),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "questions": list(rows),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input StackOverflow JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON/JSONL path")
    parser.add_argument(
        "--format",
        choices=("generator-json", "jsonl"),
        default="generator-json",
        help="Output format",
    )
    parser.add_argument("--min-score", type=int, default=None)
    parser.add_argument("--category", action="append", default=[], help="Keep only this category; repeatable")
    parser.add_argument("--allow-uncategorized", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="Start after filtering")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected rows")
    parser.add_argument("--sort-by-score", action="store_true", help="Select top rows by SO score")
    parser.add_argument("--per-category", type=int, default=None, help="Select top N rows per category")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Select this many rows close to Terminal-Bench 2.0 distribution",
    )
    parser.add_argument(
        "--distribution",
        choices=("terminal-bench-2",),
        default=None,
        help="Distribution to use with --sample-size",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.per_category is not None and args.sample_size is not None:
        raise SystemExit("--per-category and --sample-size are mutually exclusive")
    if args.sample_size is not None and args.distribution != "terminal-bench-2":
        raise SystemExit("--sample-size requires --distribution terminal-bench-2")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    rows = iter_normalized_filtered(
        args.input,
        min_score=args.min_score,
        categories=args.category,
        allow_uncategorized=args.allow_uncategorized,
    )

    if args.per_category is not None:
        selected = select_per_category(rows, args.per_category, args.category)
    elif args.sample_size is not None:
        selected = select_benchmark_distribution(rows, args.sample_size, args.category)
    elif args.sort_by_score and args.limit is not None:
        selected = select_top(rows, args.limit)
    else:
        selected = select_first(rows, args.start, args.limit)

    if args.sort_by_score and args.per_category is None and args.sample_size is None:
        selected = sorted(selected, key=sort_key_desc)
    if args.limit is not None and args.per_category is not None:
        selected = selected[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": str(args.input),
        "format": args.format,
        "min_score": args.min_score,
        "categories": args.category,
        "selection": {
            "start": args.start,
            "limit": args.limit,
            "sort_by_score": args.sort_by_score,
            "per_category": args.per_category,
            "sample_size": args.sample_size,
            "distribution": args.distribution,
        },
    }
    if args.format == "generator-json":
        write_generator_json(selected, args.output, metadata)
    else:
        write_jsonl(selected, args.output)

    category_counts: Dict[str, int] = {}
    for row in selected:
        category = row.get("selected_category") or "uncategorized"
        category_counts[category] = category_counts.get(category, 0) + 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(selected),
                "category_counts": dict(sorted(category_counts.items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

