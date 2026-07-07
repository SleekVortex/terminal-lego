from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SOQuestion:
    question_id: int
    title: str
    body: str
    tags: list[str]
    score: int
    accepted_answer_id: Optional[int]
    accepted_answer_body: Optional[str]
    accepted_answer_score: Optional[int]
    link: str
    categories: list[str]
    selected_category: Optional[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SOQuestion":
        answer = data.get("accepted_answer", {})
        categories = data.get("categories", [])
        if isinstance(categories, str):
            categories = [categories]
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        return cls(
            question_id=int(data["question_id"]),
            title=str(data["title"]),
            body=str(data.get("body") or ""),
            tags=list(tags),
            score=int(data.get("score", 0) or 0),
            accepted_answer_id=data.get("accepted_answer_id"),
            accepted_answer_body=answer.get("body") if answer else None,
            accepted_answer_score=answer.get("score") if answer else None,
            link=str(data.get("link") or ""),
            categories=list(categories),
            selected_category=data.get("selected_category") or data.get("category"),
        )


@dataclass
class TaskSpec:
    task_id: str
    question_id: int
    title: str
    category: str
    categories: list[str]
    tags: list[str]
    instruction: str
    expected_effects: list[str] = field(default_factory=list)
    required_runtimes: list[str] = field(default_factory=list)
    input_artifacts: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.task_id:
            issues.append("task_id is required")
        if not self.instruction.strip():
            issues.append("instruction is required")
        if not self.category:
            issues.append("category is required")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(data["task_id"]),
            question_id=int(data["question_id"]),
            title=str(data["title"]),
            category=str(data.get("category") or "general"),
            categories=list(data.get("categories") or []),
            tags=list(data.get("tags") or []),
            instruction=str(data.get("instruction") or ""),
            expected_effects=list(data.get("expected_effects") or []),
            required_runtimes=list(data.get("required_runtimes") or []),
            input_artifacts=list(data.get("input_artifacts") or []),
            output_artifacts=list(data.get("output_artifacts") or []),
            constraints=list(data.get("constraints") or []),
            forbidden=list(data.get("forbidden") or []),
        )


@dataclass
class EnvironmentSpec:
    files: dict[str, str] = field(default_factory=dict)
    directories: list[str] = field(default_factory=lambda: ["task_file"])

    def validate(self) -> list[str]:
        issues: list[str] = []
        for path in list(self.files) + list(self.directories):
            if path.startswith("/") or ".." in path.split("/"):
                issues.append(f"invalid relative environment path: {path}")
        if "task_file" not in {p.rstrip("/") for p in self.directories} and not any(
            p.startswith("task_file/") for p in self.files
        ):
            issues.append("environment should include task_file")
        return issues

    def to_generator_dict(self) -> dict[str, Any]:
        return {"files": self.files, "directories": self.directories}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentSpec":
        files = data.get("files") or {}
        directories = data.get("directories") or ["task_file"]
        return cls(files={str(k): str(v) for k, v in files.items()}, directories=[str(p) for p in directories])


@dataclass
class GeneratedTask:
    task_name: str
    task_spec: TaskSpec
    environment: EnvironmentSpec
    solution_sh: str
    test_outputs_py: str
    dockerfile: str
    difficulty: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "task_spec": self.task_spec.to_dict(),
            "environment": self.environment.to_generator_dict(),
            "solution_sh": self.solution_sh,
            "test_outputs_py": self.test_outputs_py,
            "dockerfile": self.dockerfile,
            "difficulty": self.difficulty,
        }


@dataclass
class ValidationResult:
    task: str
    status: str
    reward: Optional[float]
    build_time: float = 0.0
    solve_time: float = 0.0
    test_time: float = 0.0
    error: Optional[str] = None
    log_dir: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureDiagnosis:
    task_name: str
    failure_class: str
    confidence: float
    evidence: list[str]
    allowed_repairs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairAttempt:
    task_name: str
    attempt: int
    failure_class: str
    status: str
    changed_files: list[str]
    validation_result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_json_object(text: str) -> dict[str, Any]:
    return json.loads(text)
