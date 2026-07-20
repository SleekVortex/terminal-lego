"""Deep Agents Code integration for Harbor solution-generation runs.

The upstream Deep Agents repository runs its coding harness through Harbor's
installed ``langgraph`` agent. This adapter gives that setup a short
``--agent deepagent`` name, configures OpenAI-compatible local endpoints, and
converts the LangGraph result into Harbor's ATIF trajectory format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.langgraph import LangGraph
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.trajectory_utils import format_trajectory_json


DEEPAGENT_CODE_VERSION = "0.1.43"
DEEPAGENT_PROJECT_DIR = Path(__file__).with_name("deepagent_project")
DEEPAGENT_SYSTEM_PROMPT_PATH = DEEPAGENT_PROJECT_DIR / "system_prompt.txt"
DEEPAGENT_SYSTEM_PROMPT = DEEPAGENT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
DEEPAGENT_REASONING_FILENAME = "deepagent-reasoning.json"


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        if isinstance(decoded, dict):
            return {str(key): item for key, item in decoded.items()}
        return {"value": decoded}
    return {"value": value}


def _usage_metrics(value: Any, reasoning_tokens: int | None = None) -> Metrics | None:
    if not isinstance(value, dict):
        return None

    def optional_int(key: str) -> int | None:
        raw = value.get(key)
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None

    cached_tokens = None
    input_details = value.get("input_token_details")
    if isinstance(input_details, dict):
        for key in ("cache_read", "cached_tokens"):
            raw = input_details.get(key)
            if isinstance(raw, int) and not isinstance(raw, bool):
                cached_tokens = raw
                break

    prompt_tokens = optional_int("input_tokens")
    completion_tokens = optional_int("output_tokens")
    if (
        prompt_tokens is None
        and completion_tokens is None
        and cached_tokens is None
        and reasoning_tokens is None
    ):
        return None
    return Metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        extra={"reasoning_tokens": reasoning_tokens} if reasoning_tokens is not None else None,
    )


def _tool_calls(value: Any, message_index: int) -> list[ToolCall] | None:
    if not isinstance(value, list):
        return None
    calls: list[ToolCall] = []
    for call_index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            continue
        call_id = raw.get("id") or raw.get("tool_call_id")
        name = raw.get("name") or raw.get("function_name")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"deepagent-call-{message_index}-{call_index}"
        if not isinstance(name, str) or not name:
            name = "unknown_tool"
        calls.append(
            ToolCall(
                tool_call_id=call_id,
                function_name=name,
                arguments=_tool_arguments(raw.get("args", raw.get("arguments", {}))),
            )
        )
    return calls or None


def merge_reasoning_records(
    result: dict[str, Any], records_payload: dict[str, Any]
) -> int:
    """Merge the in-container reasoning sidecar into Harbor's JSON result."""
    messages = result.get("messages")
    records = records_payload.get("records")
    if not isinstance(messages, list) or not isinstance(records, list):
        return 0

    merged = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        message_index = record.get("message_index")
        reasoning = record.get("reasoning_content")
        if (
            not isinstance(message_index, int)
            or isinstance(message_index, bool)
            or not 0 <= message_index < len(messages)
            or not isinstance(reasoning, str)
            or not reasoning
        ):
            continue
        message = messages[message_index]
        if not isinstance(message, dict) or str(message.get("type") or "").lower() not in {
            "ai",
            "assistant",
        }:
            continue
        message["reasoning_content"] = reasoning
        reasoning_tokens = record.get("reasoning_tokens")
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            message["reasoning_tokens"] = reasoning_tokens
        merged += 1
    return merged


def trajectory_from_langgraph_result(
    result: dict[str, Any],
    *,
    instruction: str,
    model_name: str | None,
    session_id: str | None,
) -> Trajectory:
    """Convert Harbor LangGraph's JSON result to a best-effort ATIF trajectory."""
    steps = [
        Step(step_id=1, source="system", message=DEEPAGENT_SYSTEM_PROMPT),
        Step(step_id=2, source="user", message=instruction),
    ]
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    saw_prompt_tokens = False
    saw_completion_tokens = False
    saw_cached_tokens = False
    saw_reasoning_tokens = False
    initial_human_skipped = False

    messages = result.get("messages")
    if not isinstance(messages, list):
        messages = []

    for message_index, raw in enumerate(messages, start=1):
        if not isinstance(raw, dict):
            continue
        message_type = str(raw.get("type") or "").lower()
        content = _as_text(raw.get("content"))

        if message_type in {"human", "user"}:
            if not initial_human_skipped:
                initial_human_skipped = True
                continue
            steps.append(Step(step_id=len(steps) + 1, source="user", message=content))
            continue

        if message_type == "system":
            steps.append(Step(step_id=len(steps) + 1, source="system", message=content))
            continue

        if message_type in {"ai", "assistant"}:
            raw_reasoning_tokens = raw.get("reasoning_tokens")
            reasoning_tokens = (
                raw_reasoning_tokens
                if isinstance(raw_reasoning_tokens, int)
                and not isinstance(raw_reasoning_tokens, bool)
                else None
            )
            metrics = _usage_metrics(raw.get("usage_metadata"), reasoning_tokens)
            if metrics is not None:
                if metrics.prompt_tokens is not None:
                    total_prompt_tokens += metrics.prompt_tokens
                    saw_prompt_tokens = True
                if metrics.completion_tokens is not None:
                    total_completion_tokens += metrics.completion_tokens
                    saw_completion_tokens = True
                if metrics.cached_tokens is not None:
                    total_cached_tokens += metrics.cached_tokens
                    saw_cached_tokens = True
            if reasoning_tokens is not None:
                total_reasoning_tokens += reasoning_tokens
                saw_reasoning_tokens = True
            raw_reasoning = raw.get("reasoning_content")
            reasoning_content = (
                raw_reasoning if isinstance(raw_reasoning, str) and raw_reasoning else None
            )
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    model_name=model_name,
                    message=content,
                    reasoning_content=reasoning_content,
                    tool_calls=_tool_calls(raw.get("tool_calls"), message_index),
                    metrics=metrics,
                    llm_call_count=1,
                )
            )
            continue

        if message_type == "tool":
            target = next((step for step in reversed(steps) if step.source == "agent"), None)
            if target is None:
                target = Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    message="",
                    llm_call_count=0,
                )
                steps.append(target)

            results = target.observation.results if target.observation else []
            source_call_id = None
            if target.tool_calls and len(results) < len(target.tool_calls):
                source_call_id = target.tool_calls[len(results)].tool_call_id
            results.append(
                ObservationResult(
                    source_call_id=source_call_id,
                    content=content,
                    extra={"matched_by": "message_order"} if source_call_id else None,
                )
            )
            target.observation = Observation(results=results)

    final_metrics = FinalMetrics(
        total_prompt_tokens=total_prompt_tokens if saw_prompt_tokens else None,
        total_completion_tokens=total_completion_tokens if saw_completion_tokens else None,
        total_cached_tokens=total_cached_tokens if saw_cached_tokens else None,
        total_steps=len(steps),
        extra=(
            {"total_reasoning_tokens": total_reasoning_tokens}
            if saw_reasoning_tokens
            else None
        ),
    )
    return Trajectory(
        session_id=session_id,
        agent=Agent(
            name="deepagent",
            version=DEEPAGENT_CODE_VERSION,
            model_name=model_name,
            extra={
                "runtime": "deepagents-code",
                "harbor_agent": "langgraph",
                "graph": "deepagent",
            },
        ),
        steps=steps,
        notes=(
            "Tool-result call IDs are matched by message order because Harbor's "
            "LangGraph result serializer omits ToolMessage.tool_call_id."
        ),
        final_metrics=final_metrics,
        extra={"raw_result_path": "deepagent-result.json"},
    )


class DeepAgent(LangGraph):
    """Run Deep Agents Code through Harbor's LangGraph installed agent."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        *args: Any,
        api_base: str | None = None,
        api_key: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        configurable: dict[str, Any] | None = None,
        project_path: str | Path | None = None,
        graph: str = "deepagent",
        config: str = "langgraph.json",
        **kwargs: Any,
    ) -> None:
        merged_model_kwargs = dict(model_kwargs or {})
        model_name = kwargs.get("model_name")
        if api_base:
            merged_model_kwargs.setdefault("base_url", api_base)
            if isinstance(model_name, str) and model_name.startswith(("openai/", "openai:")):
                merged_model_kwargs.setdefault("use_responses_api", False)
        if api_key:
            merged_model_kwargs.setdefault("api_key", api_key)

        merged_configurable = {"cwd": "/app", **(configurable or {})}
        super().__init__(
            *args,
            project_path=project_path or DEEPAGENT_PROJECT_DIR,
            graph=graph,
            config=config,
            model_kwargs=merged_model_kwargs,
            configurable=merged_configurable,
            version=DEEPAGENT_CODE_VERSION,
            **kwargs,
        )

    @staticmethod
    @override
    def name() -> str:
        return "deepagent"

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        rendered_instruction = self.render_instruction(instruction)
        await super().run(instruction, environment, context)

        raw_result_path = self.logs_dir / "deepagent-result.json"
        remote_result_path = (EnvironmentPaths.agent_dir / self._RESULT_FILENAME).as_posix()
        try:
            await environment.download_file(remote_result_path, raw_result_path)
            result = json.loads(raw_result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise TypeError("DeepAgent LangGraph result must be a JSON object")

            reasoning_path = self.logs_dir / DEEPAGENT_REASONING_FILENAME
            remote_reasoning_path = (
                EnvironmentPaths.agent_dir / DEEPAGENT_REASONING_FILENAME
            ).as_posix()
            try:
                await environment.download_file(remote_reasoning_path, reasoning_path)
                reasoning_payload = json.loads(reasoning_path.read_text(encoding="utf-8"))
                if isinstance(reasoning_payload, dict):
                    merge_reasoning_records(result, reasoning_payload)
                    raw_result_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            except Exception as exc:  # noqa: BLE001 - reasoning is an optional sidecar
                self.logger.warning("Could not merge DeepAgent reasoning sidecar: %s", exc)

            trajectory = trajectory_from_langgraph_result(
                result,
                instruction=rendered_instruction,
                model_name=self.model_name,
                session_id=environment.session_id,
            )
            trajectory_path = self.logs_dir / "trajectory.json"
            trajectory_path.write_text(
                format_trajectory_json(trajectory.to_json_dict()) + "\n",
                encoding="utf-8",
            )
            context.metadata = {
                **(context.metadata or {}),
                "deepagent_raw_result": str(raw_result_path),
                "deepagent_reasoning": str(reasoning_path),
                "deepagent_trajectory": str(trajectory_path),
            }
        except Exception as exc:  # noqa: BLE001 - preserve the completed agent run
            self.logger.warning("Could not materialize DeepAgent ATIF trajectory: %s", exc)
