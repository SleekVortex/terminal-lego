from __future__ import annotations

from pathlib import Path


PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompts" / "task_generator"


def load_prompt_template(filename: str) -> str:
    return (PROMPT_TEMPLATE_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


SYSTEM_PROMPT = load_prompt_template("system.md")
INSTRUCTION_PROMPT_TEMPLATE = load_prompt_template("instruction.md")
INSTRUCTION_REWRITE_CONCISE_SYSTEM_PROMPT = load_prompt_template(
    "instruction_rewrite_concise_system.md"
)
INSTRUCTION_REWRITE_CONCISE_PROMPT_TEMPLATE = load_prompt_template(
    "instruction_rewrite_concise.md"
)
INSTRUCTION_REWRITE_LOSSY_SYSTEM_PROMPT = load_prompt_template(
    "instruction_rewrite_lossy_system.md"
)
INSTRUCTION_REWRITE_LOSSY_PROMPT_TEMPLATE = load_prompt_template(
    "instruction_rewrite_lossy.md"
)
ENVIRONMENT_PROMPT_TEMPLATE = load_prompt_template("environment.md")
SOLUTION_PROMPT_TEMPLATE = load_prompt_template("solution.md")
TEST_PROMPT_TEMPLATE = load_prompt_template("tests.md")
DOCKERFILE_PROMPT_TEMPLATE = load_prompt_template("dockerfile.md")
