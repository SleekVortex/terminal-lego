from __future__ import annotations

import hashlib


INSTRUCTION_REWRITE_MODES = ("off", "concise", "lossy", "mixed")


def validate_lossy_ratio(lossy_ratio: float) -> float:
    ratio = float(lossy_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("lossy instruction ratio must be between 0 and 1")
    return ratio


def select_instruction_style(mode: str, question_id: int, lossy_ratio: float) -> str:
    if mode not in INSTRUCTION_REWRITE_MODES:
        raise ValueError(
            f"instruction rewrite mode must be one of: {', '.join(INSTRUCTION_REWRITE_MODES)}"
        )
    ratio = validate_lossy_ratio(lossy_ratio)
    if mode != "mixed":
        return mode

    digest = hashlib.sha256(str(int(question_id)).encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "lossy" if bucket < ratio else "concise"
