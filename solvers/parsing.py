"""Parsing helpers for solver CLI options."""

from __future__ import annotations

import json
from typing import Any, Sequence


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


def parse_key_value(items: Sequence[str] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in KEY=VALUE item: {item!r}")
        values[key] = parse_scalar(raw_value.strip())
    return values


def parse_env(items: Sequence[str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in KEY=VALUE item: {item!r}")
        values[key] = raw_value.strip()
    return values
