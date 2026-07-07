from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

from generator.agents.base import LLMCall
from generator.agents.task_generation import (
    DockerfileAgent,
    EnvironmentAgent,
    InstructionAgent,
    SolutionAgent,
    TestAgent,
)
from generator.contracts import EnvironmentSpec, SOQuestion
from generator.llm_client import call_llm_api
from generator.reviewers.test_review import review_tests
from generator.task_writer import TaskWriter, format_env_file_list, generate_task_toml
from generator.text_utils import assess_difficulty


logger = logging.getLogger(__name__)


class TaskGenerator:
    def __init__(
        self,
        question: SOQuestion,
        output_dir: Path,
        index: int | None = None,
        llm_call: LLMCall = call_llm_api,
    ):
        self.question = question
        self.output_dir = output_dir
        self.llm_call = llm_call
        if index is not None:
            self.task_name = f"task_{index:05d}"
        else:
            self.task_name = self._generate_task_name()
        self.task_dir = output_dir / self.task_name
        self.guid = hashlib.md5(f"{question.question_id}".encode()).hexdigest()[:8]

    def _generate_task_name(self) -> str:
        title = self.question.title.lower()
        title = re.sub(r"[^a-z0-9\s-]", "", title)
        title = re.sub(r"\s+", "-", title)
        title = title[:50].rstrip("-")
        return f"{title}-{self.question.question_id}"

    def generate(self) -> bool:
        try:
            instruction = self._generate_instruction()
            if not instruction:
                logger.error("[%s] Failed to generate instruction", self.task_name)
                return False

            env_data = self._generate_environment(instruction)
            environment = EnvironmentSpec.from_dict(env_data)

            solution = self._generate_solution(instruction, environment.to_generator_dict())
            if not solution:
                logger.error("[%s] Failed to generate solution", self.task_name)
                return False

            difficulty = self._assess_difficulty(instruction)

            test_data = self._generate_tests(instruction, environment.to_generator_dict(), solution)
            if not test_data:
                logger.error("[%s] Failed to generate valid tests", self.task_name)
                return False

            dockerfile = self._generate_dockerfile(instruction, environment.to_generator_dict(), solution, test_data)
            if not dockerfile:
                logger.error("[%s] Failed to generate valid Dockerfile", self.task_name)
                return False

            TaskWriter(self.task_dir, self.question).write(
                instruction,
                environment,
                test_data.get("test_outputs_py", ""),
                solution,
                dockerfile,
                difficulty,
            )
            logger.info("[%s] Written to: %s", self.task_name, self.task_dir)
            return True
        except Exception as exc:
            logger.exception("[%s] Exception: %s", self.task_name, exc)
            return False

    def _generate_instruction(self) -> Optional[str]:
        return InstructionAgent(self.llm_call, self.task_name).run(self.question)

    def _generate_environment(self, instruction: str) -> dict[str, Any]:
        return EnvironmentAgent(self.llm_call, self.task_name).run(self.question, instruction)

    def _format_env_file_list(self, env_data: dict[str, Any]) -> str:
        return format_env_file_list(env_data)

    def _generate_solution(self, instruction: str, env_data: dict[str, Any]) -> Optional[str]:
        return SolutionAgent(self.llm_call, self.task_name).run(self.question, instruction, env_data)

    def _generate_tests(self, instruction: str, env_data: dict[str, Any], solution: str) -> Optional[dict[str, str]]:
        return TestAgent(self.llm_call, self.task_name).run(self.question, instruction, env_data, solution)

    def _review_tests(self, instruction: str, env_file_list: str, solution: str, test_code: str) -> dict[str, object]:
        del instruction, env_file_list, solution
        return review_tests(test_code)

    def _generate_dockerfile(
        self,
        instruction: str,
        env_data: Optional[dict[str, Any]] = None,
        solution: str = "",
        test_data: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        return DockerfileAgent(self.llm_call, self.task_name).run(
            self.question,
            instruction,
            env_data,
            solution,
            test_data,
        )

    def _assess_difficulty(self, instruction: str) -> str:
        del instruction
        return assess_difficulty(self.question.body, self.question.accepted_answer_body)

    def _generate_task_toml(self, difficulty: str = "medium") -> str:
        return generate_task_toml(self.question, difficulty)

    def _write_files(
        self,
        instruction: str,
        env_data: dict[str, Any],
        test_data: dict[str, str],
        solution: str,
        dockerfile: str,
        difficulty: str = "medium",
    ) -> None:
        TaskWriter(self.task_dir, self.question).write(
            instruction,
            EnvironmentSpec.from_dict(env_data),
            test_data.get("test_outputs_py", ""),
            solution,
            dockerfile,
            difficulty,
        )
