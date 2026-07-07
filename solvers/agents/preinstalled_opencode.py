"""Harbor OpenCode agent that uses a pre-mounted OpenCode runtime.

The stock Harbor OpenCode agent installs nvm, Node, and opencode-ai inside each
task container. That is too slow and depends on GitHub/npm availability during
every trial. This variant keeps Harbor's OpenCode run/trajectory logic, but
replaces installation with a local runtime check.
"""

from __future__ import annotations

import os
from typing import Any

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.trajectories import Step, Trajectory


OPENCODE_SYSTEM_PROMPT = """You are OpenCode's build agent running inside a Harbor/Terminal-Lego task container.

Use the available OpenCode tools to solve the user's terminal task in the container filesystem.
Read and modify task files as needed, run shell commands when useful, and prefer deterministic,
non-interactive, idempotent changes.

For Terminal-Lego solution generation, before the final response make sure an executable
reproduction script exists at /logs/artifacts/solve.sh. The script must be non-interactive,
idempotent, and must reproduce the successful task changes from a fresh task image without
reading /tests, /solution, /logs/verifier, reward files, or any hidden reference solution."""


class PreinstalledOpenCode(OpenCode):
    """Use OpenCode from /opt/terminal-lego/opencode instead of downloading it."""

    _DEFAULT_CONFIG = {
        "agent": {
            "title": {"disable": True},
        },
    }

    _GLM_PROVIDER_ID = "glm"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._original_model_name = self.model_name
        if self.model_name and "/" in self.model_name:
            provider, model_id = self.model_name.split("/", 1)
            if provider == "openai" and model_id.lower().startswith("glm"):
                self.model_name = f"{self._GLM_PROVIDER_ID}/{model_id}"

    def _build_register_config_command(self) -> str | None:
        if self.model_name and "/" in self.model_name:
            provider, model_id = self.model_name.split("/", 1)
            if provider == self._GLM_PROVIDER_ID:
                base_url = (
                    os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("OPENAI_API_BASE")
                    or "http://host.docker.internal:30003/v1"
                )
                api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
                self._opencode_config = self._deep_merge(
                    {
                        "model": self.model_name,
                        "small_model": self.model_name,
                        "agent": {
                            "build": {
                                "model": self.model_name,
                                "prompt": OPENCODE_SYSTEM_PROMPT,
                            },
                        },
                        "provider": {
                            self._GLM_PROVIDER_ID: {
                                "npm": "@ai-sdk/openai-compatible",
                                "name": "GLM local",
                                "options": {
                                    "baseURL": base_url,
                                    "apiKey": api_key,
                                    "timeout": 600000,
                                    "headerTimeout": 600000,
                                    "chunkTimeout": 600000,
                                },
                                "models": {
                                    model_id: {
                                        "name": model_id,
                                        "reasoning": True,
                                        "temperature": True,
                                        "tool_call": True,
                                        "interleaved": {
                                            "field": "reasoning_content",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    self._opencode_config,
                )
        return super()._build_register_config_command()

    def _convert_events_to_trajectory(
        self, events: list[dict[str, Any]]
    ) -> Trajectory | None:
        trajectory = super()._convert_events_to_trajectory(events)
        if trajectory is None:
            return None

        if not any(step.source == "system" for step in trajectory.steps):
            trajectory.steps.insert(
                0,
                Step(
                    step_id=1,
                    source="system",
                    message=OPENCODE_SYSTEM_PROMPT,
                    extra={
                        "origin": "terminal-lego.preinstalled-opencode",
                        "prompt_config": "agent.build.prompt",
                    },
                ),
            )
            for index, step in enumerate(trajectory.steps, start=1):
                step.step_id = index
            if trajectory.final_metrics:
                trajectory.final_metrics.total_steps = len(trajectory.steps)

        return trajectory

    @staticmethod
    def name() -> str:
        return "preinstalled-opencode"

    def get_version_command(self) -> str | None:
        return "opencode --version"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                "test -x /opt/terminal-lego/opencode/bin/opencode; "
                "ln -sf /opt/terminal-lego/opencode/bin/opencode /usr/local/bin/opencode; "
                "command -v opencode; "
                "opencode --version"
            ),
        )
        await self.exec_as_agent(
            environment,
            # Harbor's OpenCode.run sources ~/.nvm/nvm.sh before invoking opencode.
            # The preinstalled runtime does not need nvm, so provide a no-op file.
            command="mkdir -p ~/.nvm && touch ~/.nvm/nvm.sh",
        )
