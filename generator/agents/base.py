from __future__ import annotations

from typing import Callable, Optional


LLMCall = Callable[..., Optional[str]]


class LLMAgent:
    def __init__(self, llm_call: LLMCall, system_prompt: str, task_name: str):
        self.llm_call = llm_call
        self.system_prompt = system_prompt
        self.task_name = task_name

    def call(self, prompt: str, stage: str, check_truncation: bool = False) -> Optional[str]:
        return self.llm_call(
            prompt,
            self.system_prompt,
            temperature=None,
            check_truncation=check_truncation,
            task_name=self.task_name,
            stage=stage,
        )
