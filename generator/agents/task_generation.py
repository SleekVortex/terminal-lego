from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from generator.agents.base import LLMCall, LLMAgent
from generator.contracts import SOQuestion
from generator.prompt_loader import (
    DOCKERFILE_PROMPT_TEMPLATE,
    ENVIRONMENT_PROMPT_TEMPLATE,
    INSTRUCTION_PROMPT_TEMPLATE,
    SOLUTION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    TEST_PROMPT_TEMPLATE,
)
from generator.reviewers.dockerfile_review import (
    ensure_verifier_deps,
    extract_dockerfile_from_response,
    review_dockerfile,
)
from generator.reviewers.solution_review import review_solution_shell
from generator.reviewers.test_review import STATIC_TEST_SH, parse_test_outputs_py, review_tests
from generator.settings import DOCKERFILE_MAX_ATTEMPTS, TEST_MAX_ATTEMPTS
from generator.task_writer import format_env_file_list
from generator.text_utils import clean_html


logger = logging.getLogger(__name__)


def extract_fenced(response: str, language: str) -> Optional[str]:
    match = re.search(rf"```{language}\n?(.*?)```", response.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_json_response(response: str) -> Optional[dict[str, Any]]:
    match = re.search(r"```json\n?(.*?)```", response, re.DOTALL)
    candidates = [match.group(1)] if match else []
    candidates.append(response)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class TaskStageAgent(LLMAgent):
    def __init__(self, llm_call: LLMCall, task_name: str):
        super().__init__(llm_call, SYSTEM_PROMPT, task_name)


class InstructionAgent(TaskStageAgent):
    def run(self, question: SOQuestion) -> Optional[str]:
        prompt = INSTRUCTION_PROMPT_TEMPLATE.format(
            title=question.title,
            tags=", ".join(question.tags),
            body=clean_html(question.body),
        )
        response = self.call(prompt, "instruction", check_truncation=True)
        if not response:
            return None
        response = response.strip()
        fenced = extract_fenced(response, "markdown")
        if fenced is not None:
            return fenced
        if response.startswith("```markdown\n"):
            response = response[12:]
            if response.endswith("```"):
                response = response[:-3]
        return response.strip()


class EnvironmentAgent(TaskStageAgent):
    def run(self, question: SOQuestion, instruction: str) -> dict[str, Any]:
        prompt = ENVIRONMENT_PROMPT_TEMPLATE.format(
            instruction=instruction,
            title=question.title,
            tags=", ".join(question.tags),
        )
        response = self.call(prompt, "environment")
        if not response:
            return {"files": {}, "directories": ["task_file"]}
        parsed = parse_json_response(response)
        if parsed is None:
            return {"files": {}, "directories": ["task_file"]}
        return parsed


class SolutionAgent(TaskStageAgent):
    def run(self, question: SOQuestion, instruction: str, env_data: dict[str, Any]) -> Optional[str]:
        answer = clean_html(question.accepted_answer_body) if question.accepted_answer_body else ""
        prompt = SOLUTION_PROMPT_TEMPLATE.format(
            instruction=instruction,
            answer=answer,
            tags=", ".join(question.tags),
            env_file_list=format_env_file_list(env_data),
        )
        response = self.call(prompt, "solution")
        if not response:
            return None
        solution = extract_fenced(response, "bash") or extract_fenced(response, "sh") or response.strip()
        review = review_solution_shell(solution)
        if not review.get("pass"):
            logger.warning("[%s] solution static review rejected: %s", self.task_name, review.get("issues"))
        return solution


class TestAgent(TaskStageAgent):
    def run(self, question: SOQuestion, instruction: str, env_data: dict[str, Any], solution: str) -> Optional[dict[str, str]]:
        env_file_list = format_env_file_list(env_data)
        candidates: list[tuple[dict[str, str], list[str]]] = []
        for attempt in range(TEST_MAX_ATTEMPTS):
            prompt = TEST_PROMPT_TEMPLATE.format(
                instruction=instruction,
                env_file_list=env_file_list,
                solution=solution,
                tags=", ".join(question.tags),
            )
            response = self.call(prompt, "tests")
            if not response:
                continue
            parsed = parse_json_response(response)
            if not parsed:
                continue
            test_py = str(parsed.get("test_outputs_py") or "")
            if not test_py:
                continue
            try:
                parse_test_outputs_py(test_py)
            except SyntaxError as exc:
                logger.warning("[%s] ast.parse failed (round %s): %s", self.task_name, attempt + 1, exc.msg)
                continue
            review = review_tests(test_py)
            if review.get("pass"):
                logger.info("[%s] test review passed (round %s)", self.task_name, attempt + 1)
                return {"test_sh": STATIC_TEST_SH, "test_outputs_py": test_py}
            issues = list(review.get("issues", []))
            logger.warning("[%s] test review rejected (round %s): %s", self.task_name, attempt + 1, issues)
            candidates.append(({"test_sh": STATIC_TEST_SH, "test_outputs_py": test_py}, issues))
        if candidates:
            return candidates[0][0]
        return None


class DockerfileAgent(TaskStageAgent):
    def run(
        self,
        question: SOQuestion,
        instruction: str,
        env_data: Optional[dict[str, Any]] = None,
        solution: str = "",
        test_data: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        env_data = env_data or {"files": {}, "directories": ["task_file"]}
        test_outputs_py = test_data.get("test_outputs_py", "") if test_data else ""
        base_prompt = DOCKERFILE_PROMPT_TEMPLATE.format(
            instruction=instruction,
            tags=", ".join(question.tags),
            env_file_list=format_env_file_list(env_data),
            solution=solution,
            test_outputs_py=test_outputs_py,
        )
        feedback = ""
        for attempt in range(DOCKERFILE_MAX_ATTEMPTS):
            prompt = base_prompt
            if feedback:
                prompt += (
                    "\n\nPrevious Dockerfile attempt was rejected for these issues:\n"
                    + feedback
                    + "\nRegenerate only the Dockerfile."
                )
            response = self.call(prompt, "dockerfile")
            if not response:
                feedback = "- LLM returned no Dockerfile response"
                continue
            extracted = extract_dockerfile_from_response(response)
            if not extracted:
                feedback = "- Dockerfile response did not contain a fenced Dockerfile block"
                continue
            dockerfile = ensure_verifier_deps(extracted)
            review = review_dockerfile(dockerfile, env_data)
            if review.get("pass"):
                return dockerfile
            issues = list(review.get("issues", []))
            feedback = "\n".join(f"- {issue}" for issue in issues)
            logger.warning("[%s] Dockerfile review rejected (round %s): %s", self.task_name, attempt + 1, issues)
        return None
