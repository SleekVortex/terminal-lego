#!/usr/bin/env python3
"""Terminal-Bench category taxonomy for StackOverflow rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


TAXONOMY_PATH = Path(__file__).with_name("terminal_bench_tag_taxonomy.json")


def _load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_TAXONOMY = _load_taxonomy()

TAGS_BY_CATEGORY: Dict[str, List[str]] = {
    category: list(tags)
    for category, tags in _TAXONOMY["categories"].items()
}

TERMINAL_BENCH_2_DISTRIBUTION: Dict[str, int] = {
    category: int(count)
    for category, count in _TAXONOMY["benchmark_distribution"].items()
}

CATEGORY_ORDER: List[str] = list(TERMINAL_BENCH_2_DISTRIBUTION)


def normalize_tag(tag: str) -> str:
    return str(tag).strip().lower()


def classify_tags(tags: Iterable[str]) -> List[str]:
    """Return all Terminal-Bench categories that match the given SO tags."""
    tag_set = {normalize_tag(tag) for tag in tags if str(tag).strip()}
    categories = []
    for category in CATEGORY_ORDER:
        category_tags = {normalize_tag(tag) for tag in TAGS_BY_CATEGORY[category]}
        if tag_set & category_tags:
            categories.append(category)
    return categories


def category_tag_hits(tags: Iterable[str]) -> Dict[str, List[str]]:
    """Return matched tags by category for audit/debug output."""
    tag_set = {normalize_tag(tag) for tag in tags if str(tag).strip()}
    hits: Dict[str, List[str]] = {}
    for category in CATEGORY_ORDER:
        matched = sorted(
            tag_set & {normalize_tag(tag) for tag in TAGS_BY_CATEGORY[category]}
        )
        if matched:
            hits[category] = matched
    return hits


def scaled_quotas(
    total: int,
    distribution: Mapping[str, int] = TERMINAL_BENCH_2_DISTRIBUTION,
) -> Dict[str, int]:
    """Scale integer benchmark counts to an exact integer total."""
    if total <= 0:
        return {category: 0 for category in distribution}

    base_total = sum(distribution.values())
    raw = {
        category: total * count / base_total
        for category, count in distribution.items()
    }
    quotas = {category: int(value) for category, value in raw.items()}
    remainder = total - sum(quotas.values())

    order = sorted(
        raw,
        key=lambda category: (raw[category] - quotas[category], raw[category]),
        reverse=True,
    )
    for category in order[:remainder]:
        quotas[category] += 1
    return quotas

