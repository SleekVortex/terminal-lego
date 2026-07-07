from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from generator.agents.base import LLMCall, LLMAgent
from generator.agents.task_generation import parse_json_response
from generator.contracts import FailureDiagnosis
from generator.prompt_loader import SYSTEM_PROMPT
from generator.reviewers.dockerfile_review import ensure_verifier_deps, review_dockerfile
from generator.reviewers.solution_review import review_solution_shell
from generator.reviewers.test_review import parse_test_outputs_py, review_tests


PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "repair" / "task_repair.md"
MAX_FILE_CHARS = 12000


def read_text(path: Path, limit: int = MAX_FILE_CHARS) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text


def safe_relative_path(path: str) -> bool:
    parts = Path(path).parts
    return bool(path) and not path.startswith("/") and ".." not in parts


def normalize_allowed_repairs(items: list[str]) -> set[str]:
    allowed: set[str] = set()
    for item in items:
        normalized = item.strip().strip("/")
        if normalized in {"Dockerfile", "environment/Dockerfile"}:
            allowed.add("environment/Dockerfile")
        elif normalized in {"solution", "solution/solve.sh"}:
            allowed.add("solution/solve.sh")
        elif normalized in {"tests", "tests/test_outputs.py", "test_outputs.py"}:
            allowed.add("tests/test_outputs.py")
        elif normalized == "environment":
            allowed.add("environment/**")
        elif safe_relative_path(normalized):
            allowed.add(normalized)
    return allowed


def is_allowed_file(path: str, allowed: set[str]) -> bool:
    if not safe_relative_path(path):
        return False
    if path in allowed:
        return True
    return "environment/**" in allowed and path.startswith("environment/")


def collect_task_files(task_dir: Path) -> dict[str, str]:
    files = {
        "instruction.md": read_text(task_dir / "instruction.md"),
        "environment/Dockerfile": read_text(task_dir / "environment" / "Dockerfile"),
        "solution/solve.sh": read_text(task_dir / "solution" / "solve.sh"),
        "tests/test_outputs.py": read_text(task_dir / "tests" / "test_outputs.py"),
    }
    env_dir = task_dir / "environment"
    if env_dir.exists():
        for path in sorted(env_dir.rglob("*")):
            if path.is_file() and path.name != "Dockerfile":
                rel = path.relative_to(task_dir).as_posix()
                if len(files) >= 12:
                    break
                files[rel] = read_text(path, limit=4000)
    return files


def env_data_from_task(task_dir: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    directories: list[str] = []
    env_dir = task_dir / "environment"
    if not env_dir.exists():
        return {"files": {}, "directories": ["task_file"]}
    for path in sorted(env_dir.rglob("*")):
        rel = path.relative_to(env_dir).as_posix()
        if rel == "Dockerfile":
            continue
        if path.is_dir():
            directories.append(rel)
        elif path.is_file():
            files[rel] = read_text(path, limit=4000)
    if "task_file" not in {p.rstrip("/") for p in directories} and not any(p.startswith("task_file/") for p in files):
        directories.insert(0, "task_file")
    return {"files": files, "directories": directories}


class RepairAgent(LLMAgent):
    def __init__(self, llm_call: LLMCall, task_name: str):
        super().__init__(llm_call, SYSTEM_PROMPT, task_name)
        self.prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    def repair(self, source_task_dir: Path, output_task_dir: Path, diagnosis: FailureDiagnosis) -> tuple[list[str], str]:
        if output_task_dir.exists():
            shutil.rmtree(output_task_dir)
        shutil.copytree(source_task_dir, output_task_dir)

        allowed = normalize_allowed_repairs(diagnosis.allowed_repairs)
        if not allowed:
            return [], "no allowed repairs"

        prompt = self.prompt_template.format(
            task_name=diagnosis.task_name,
            failure_class=diagnosis.failure_class,
            evidence="\n\n".join(diagnosis.evidence),
            allowed_files="\n".join(f"- {path}" for path in sorted(allowed)),
            task_files=json.dumps(collect_task_files(output_task_dir), indent=2, ensure_ascii=False),
        )
        response = self.call(prompt, "repair")
        if not response:
            return [], "LLM returned no repair response"
        parsed = parse_json_response(response)
        if not parsed or not isinstance(parsed.get("files"), dict):
            return [], "repair response did not contain files object"

        changed_files: list[str] = []
        for rel_path, content in parsed["files"].items():
            rel_path = str(rel_path)
            if not is_allowed_file(rel_path, allowed):
                return changed_files, f"repair tried to edit non-allowed file: {rel_path}"
            new_text = str(content)
            reviewed_text, issue = self._review_changed_file(output_task_dir, rel_path, new_text)
            if issue:
                return changed_files, issue
            target = output_task_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(reviewed_text.rstrip() + "\n", encoding="utf-8")
            if rel_path == "solution/solve.sh":
                target.chmod(0o755)
            changed_files.append(rel_path)
        return changed_files, str(parsed.get("notes") or "")

    def _review_changed_file(self, task_dir: Path, rel_path: str, content: str) -> tuple[str, Optional[str]]:
        if rel_path == "environment/Dockerfile":
            dockerfile = ensure_verifier_deps(content)
            review = review_dockerfile(dockerfile, env_data_from_task(task_dir))
            if not review.get("pass"):
                return dockerfile, "Dockerfile repair rejected: " + "; ".join(map(str, review.get("issues", [])))
            return dockerfile, None
        if rel_path == "solution/solve.sh":
            review = review_solution_shell(content)
            if not review.get("pass"):
                return content, "solution repair rejected: " + "; ".join(map(str, review.get("issues", [])))
            return content, None
        if rel_path == "tests/test_outputs.py":
            try:
                parse_test_outputs_py(content)
            except SyntaxError as exc:
                return content, f"test repair syntax error: {exc.msg}"
            review = review_tests(content)
            if not review.get("pass"):
                return content, "test repair rejected: " + "; ".join(map(str, review.get("issues", [])))
            return content, None
        if rel_path.startswith("environment/"):
            return content, None
        return content, f"unsupported repair target: {rel_path}"
