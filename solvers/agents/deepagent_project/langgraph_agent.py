"""LangGraph factory for running Deep Agents Code under Harbor."""

from __future__ import annotations

import os
import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_code.agent import create_cli_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.language_models import BaseChatModel


_DEFAULT_WORKDIR = Path("/app")
_SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
_REASONING_PATH = Path("/logs/agent/deepagent-reasoning.json")
_SHELL_ENV_DENYLIST = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    }
)


@contextmanager
def _scrub_shell_env() -> Iterator[None]:
    saved = {name: os.environ.pop(name, None) for name in _SHELL_ENV_DENYLIST}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _configurable(config: dict[str, object] | None) -> dict[str, object]:
    if config is None:
        return {}
    value = config.get("configurable")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("`configurable` must be a dictionary")
    return {str(key): item for key, item in value.items()}


def _model_kwargs(configurable: dict[str, object]) -> dict[str, Any]:
    value = configurable.get("model_kwargs")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("`configurable.model_kwargs` must be a dictionary")
    return {str(key): item for key, item in value.items()}


def _model_name(configurable: dict[str, object]) -> str:
    value = configurable.get("model") or os.environ.get("HARBOR_MODEL")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`configurable.model` or `HARBOR_MODEL` must provide a model name")
    return value


class _ReasoningChatOpenAI(ChatOpenAI):
    """Keep OpenAI-compatible ``reasoning_content`` on LangChain messages."""

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> Any:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(warnings=False)
        )
        choices = response_dict.get("choices")
        if not isinstance(choices, list):
            return result

        usage = response_dict.get("usage")
        reasoning_tokens = usage.get("reasoning_tokens") if isinstance(usage, dict) else None
        for choice, generation in zip(choices, result.generations, strict=False):
            if not isinstance(choice, dict):
                continue
            raw_message = choice.get("message")
            if not isinstance(raw_message, dict):
                continue
            reasoning = raw_message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
            if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
                generation.message.additional_kwargs["reasoning_tokens"] = reasoning_tokens
        return result


def _build_model(configurable: dict[str, object]) -> BaseChatModel:
    name = _model_name(configurable)
    kwargs = _model_kwargs(configurable)
    if name.startswith("openai:") and "use_responses_api" not in kwargs:
        kwargs["use_responses_api"] = True
    if name.startswith("openai:"):
        return _ReasoningChatOpenAI(model=name.split(":", 1)[1], **kwargs)
    return init_chat_model(name, **kwargs)


def _workdir(configurable: dict[str, object]) -> Path:
    value = configurable.get("cwd")
    if value is None:
        return _DEFAULT_WORKDIR
    if not isinstance(value, str | Path):
        raise TypeError("`configurable.cwd` must be a string path")
    return Path(value)


def _write_reasoning_sidecar(result: Any) -> None:
    messages = result.get("messages") if isinstance(result, dict) else None
    records = []
    for message_index, message in enumerate(messages or []):
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if not isinstance(additional_kwargs, dict):
            continue
        reasoning = additional_kwargs.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            continue
        record: dict[str, Any] = {
            "message_index": message_index,
            "reasoning_content": reasoning,
        }
        reasoning_tokens = additional_kwargs.get("reasoning_tokens")
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            record["reasoning_tokens"] = reasoning_tokens
        records.append(record)

    _REASONING_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REASONING_PATH.write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class _ReasoningCaptureGraph:
    """Write reasoning after the graph completes without changing its messages."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    async def ainvoke(
        self,
        input_value: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> Any:
        result = await self._graph.ainvoke(input_value, config=config)
        _write_reasoning_sidecar(result)
        return result


def make_graph(config: dict[str, object] | None = None) -> object:
    """Build the headless Deep Agents Code graph used by Harbor."""
    configurable = _configurable(config)
    model = _build_model(configurable)
    assistant_id = os.environ.get("HARBOR_SESSION_ID") or f"harbor-{uuid.uuid4()}"
    system_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    with _scrub_shell_env():
        graph, _backend = create_cli_agent(
            model=model,
            assistant_id=assistant_id,
            sandbox=None,
            sandbox_type="harbor",
            system_prompt=system_prompt,
            interactive=False,
            auto_approve=True,
            enable_memory=False,
            enable_skills=False,
            enable_shell=True,
            cwd=_workdir(configurable),
        )
    return _ReasoningCaptureGraph(graph)
