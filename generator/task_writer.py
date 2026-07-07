from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from generator.contracts import EnvironmentSpec, SOQuestion
from generator.reviewers.dockerfile_review import ensure_verifier_deps
from generator.reviewers.test_review import STATIC_TEST_SH


def toml_quote(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def task_category(question: SOQuestion) -> str:
    category_mapping = {
        "linux": "system-administration",
        "bash": "shell-scripting",
        "python": "programming",
        "git": "version-control",
        "docker": "containerization",
        "nginx": "web-server",
        "ssh": "networking",
        "networking": "networking",
        "database": "database",
        "sql": "database",
    }
    if question.selected_category:
        return question.selected_category
    if question.categories:
        return question.categories[0]
    for tag in question.tags:
        if tag.lower() in category_mapping:
            return category_mapping[tag.lower()]
    return "general"


def generate_task_toml(question: SOQuestion, difficulty: str = "medium") -> str:
    tags_str = ", ".join(toml_quote(tag) for tag in question.tags[:5])
    categories_str = ", ".join(toml_quote(category) for category in question.categories)
    return f'''version = "1.0"

[metadata]
author_name = "StackOverflow Community"
author_email = "community@stackoverflow.com"
difficulty = {toml_quote(difficulty)}
category = {toml_quote(task_category(question))}
tags = [{tags_str}]
categories = [{categories_str}]
source_question_id = {int(question.question_id)}
source_url = {toml_quote(question.link)}
source_score = {question.score}

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 600.0

[environment]
build_timeout_sec = 120.0
cpus = 1
memory = "1G"
storage = "5G"
'''


def format_env_file_list(env_data: dict[str, Any]) -> str:
    lines: list[str] = []
    for dir_path in env_data.get("directories", []):
        lines.append(f"  [dir]  /app/{dir_path}/")
    for file_path, content in env_data.get("files", {}).items():
        preview = content[:200] + ("..." if len(content) > 200 else "")
        lines.append(f"  [file] /app/{file_path}")
        lines.append(f"         Content: {preview}")
    return "\n".join(lines) if lines else "  (no environment files)"


class TaskWriter:
    def __init__(self, task_dir: Path, question: SOQuestion):
        self.task_dir = task_dir
        self.question = question

    def write(
        self,
        instruction: str,
        environment: EnvironmentSpec,
        test_outputs_py: str,
        solution: str,
        dockerfile: str,
        difficulty: str = "medium",
    ) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        (self.task_dir / "environment").mkdir(exist_ok=True)
        (self.task_dir / "tests").mkdir(exist_ok=True)
        (self.task_dir / "solution").mkdir(exist_ok=True)

        dockerfile = ensure_verifier_deps(dockerfile)

        (self.task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
        (self.task_dir / "task.toml").write_text(generate_task_toml(self.question, difficulty), encoding="utf-8")
        (self.task_dir / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")

        for dir_path in environment.directories:
            (self.task_dir / "environment" / dir_path).mkdir(parents=True, exist_ok=True)
        for file_path, content in environment.files.items():
            full_path = self.task_dir / "environment" / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")

        test_sh_path = self.task_dir / "tests" / "test.sh"
        solve_sh_path = self.task_dir / "solution" / "solve.sh"
        (self.task_dir / "tests" / "__init__.py").write_text("", encoding="utf-8")
        test_sh_path.write_text(STATIC_TEST_SH, encoding="utf-8")
        (self.task_dir / "tests" / "test_outputs.py").write_text(test_outputs_py, encoding="utf-8")
        solve_sh_path.write_text(solution, encoding="utf-8")
        test_sh_path.chmod(0o755)
        solve_sh_path.chmod(0o755)
