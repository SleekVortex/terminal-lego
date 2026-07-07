from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "configs" / "opencode" / "openai_compat_proxy.py"

spec = importlib.util.spec_from_file_location("openai_compat_proxy", MODULE_PATH)
assert spec is not None
proxy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(proxy)


def test_normalize_chat_completions_request_removes_ai_sdk_fields() -> None:
    body = {
        "model": "glm-5.2-fp8",
        "messages": [{"role": "user", "content": "ok"}],
        "stream": True,
        "reasoning": {"effort": "none"},
        "providerOptions": {"openai": {"reasoningEffort": "none"}},
        "metadata": {"source": "opencode"},
        "max_completion_tokens": 32,
        "chat_template_kwargs": {"foo": "bar"},
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }

    normalized = proxy.normalize_chat_completions_request(body)

    assert normalized["model"] == "glm-5.2-fp8"
    assert normalized["messages"] == body["messages"]
    assert normalized["tools"] == body["tools"]
    assert normalized["max_tokens"] == 32
    assert "max_completion_tokens" not in normalized
    assert "reasoning" not in normalized
    assert "providerOptions" not in normalized
    assert "metadata" not in normalized
    assert normalized["chat_template_kwargs"] == {"foo": "bar"}


def test_disable_thinking_is_explicit_fallback() -> None:
    normalized = proxy.normalize_chat_completions_request(
        {
            "model": "glm-5.2-fp8",
            "messages": [{"role": "user", "content": "ok"}],
        },
        disable_thinking=True,
    )

    assert normalized["chat_template_kwargs"] == {"enable_thinking": False}


def test_normalize_request_only_rewrites_chat_completions() -> None:
    body = json.dumps(
        {
            "model": "glm-5.2-fp8",
            "messages": [{"role": "user", "content": "ok"}],
            "reasoning": {"effort": "none"},
        }
    ).encode()

    rewritten = proxy.normalize_request("/v1/chat/completions", body)
    assert rewritten != body
    parsed = json.loads(rewritten)
    assert "reasoning" not in parsed
    assert "chat_template_kwargs" not in parsed

    assert proxy.normalize_request("/v1/models", body) == body


def test_prepare_upstream_request_rewrites_responses_to_chat_completions() -> None:
    body = json.dumps(
        {
            "model": "glm-5.2-fp8",
            "input": [
                {"role": "system", "content": "system prompt"},
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
            "stream": True,
            "tool_choice": "auto",
            "tools": [
                {
                    "type": "function",
                    "name": "bash",
                    "description": "run command",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
    ).encode()

    target_path, rewritten, response_mode = proxy.prepare_upstream_request(
        "/v1/responses", body
    )
    parsed = json.loads(rewritten)

    assert target_path == "/v1/chat/completions"
    assert response_mode == proxy.RESPONSE_MODE_OPENAI_RESPONSES
    assert parsed["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]
    assert parsed["tool_choice"] == "auto"
    assert parsed["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run command",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_responses_stream_adapter_converts_chat_events() -> None:
    adapter = proxy.ResponsesStreamAdapter()

    first = adapter.convert_line(
        b'data: {"id":"chatcmpl_1","created":1,"model":"glm",'
        b'"choices":[{"delta":{"reasoning_content":"think"}}]}\n'
    )
    second = adapter.convert_line(
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
    )
    done = adapter.convert_line(b"data: [DONE]\n")
    combined = first + second + done

    assert b'"type":"response.created"' in combined
    assert b'"type":"response.output_item.added"' in combined
    assert b'"type":"response.reasoning_text.delta"' in combined
    assert b'"type":"response.output_text.delta"' in combined
    assert b'"type":"response.completed"' in combined
    assert b"data: [DONE]" in combined


def test_normalize_sse_line_aliases_reasoning_content() -> None:
    line = (
        b'data: {"choices":[{"delta":{"reasoning_content":"think",'
        b'"content":""}}]}\n'
    )

    normalized = proxy.normalize_sse_line(line)
    payload = json.loads(normalized.removeprefix(b"data: ").decode("utf-8"))

    assert payload["choices"][0]["delta"]["reasoning_content"] == "think"
    assert payload["choices"][0]["delta"]["reasoning"] == "think"


def test_build_target_url_preserves_v1_prefix() -> None:
    assert (
        proxy.build_target_url("http://127.0.0.1:30003", "/v1/models")
        == "http://127.0.0.1:30003/v1/models"
    )
    assert (
        proxy.build_target_url("http://127.0.0.1:30003/v1", "/v1/models")
        == "http://127.0.0.1:30003/v1/models"
    )
