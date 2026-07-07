#!/usr/bin/env python3
"""Small OpenAI-compatible request normalizer for OpenCode -> GLM.

OpenCode uses Vercel AI SDK under the hood. Some AI SDK request fields are not
accepted by the GLM OpenAI-compatible endpoint. GLM 5.2 streams thinking in
delta.reasoning_content, while OpenCode/AI SDK expects a more standard reasoning
field. This proxy keeps the public OpenAI route shape, rewrites only chat
requests, and aliases response reasoning without disabling it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

CHAT_COMPLETIONS_KEYS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "model",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "chat_template_kwargs",
}

RESPONSE_MODE_PASSTHROUGH = "passthrough"
RESPONSE_MODE_OPENAI_RESPONSES = "openai_responses"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def normalize_chat_completions_request(
    body: dict[str, Any], *, disable_thinking: bool = False
) -> dict[str, Any]:
    """Return a GLM-compatible chat/completions request body."""
    normalized = {
        key: value for key, value in body.items() if key in CHAT_COMPLETIONS_KEYS
    }

    if "max_completion_tokens" in normalized and "max_tokens" not in normalized:
        normalized["max_tokens"] = normalized["max_completion_tokens"]
    normalized.pop("max_completion_tokens", None)

    if disable_thinking:
        chat_template_kwargs = normalized.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        else:
            chat_template_kwargs = dict(chat_template_kwargs)
        chat_template_kwargs["enable_thinking"] = False
        normalized["chat_template_kwargs"] = chat_template_kwargs
    elif "chat_template_kwargs" in normalized and not isinstance(
        normalized["chat_template_kwargs"], dict
    ):
        normalized.pop("chat_template_kwargs", None)

    return normalized


def normalize_request(
    path: str, body: bytes | None, *, disable_thinking: bool = False
) -> bytes | None:
    if not body:
        return body

    route = urllib.parse.urlsplit(path).path.rstrip("/")
    if route not in {"/v1/chat/completions", "/chat/completions"}:
        return body

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body

    if not isinstance(payload, dict):
        return body

    normalized = normalize_chat_completions_request(
        payload, disable_thinking=disable_thinking
    )
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def content_parts_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if text:
                texts.append(str(text))
        elif part_type == "refusal":
            refusal = part.get("refusal")
            if refusal:
                texts.append(str(refusal))
    return "\n".join(texts)


def responses_input_to_messages(input_items: Any) -> list[dict[str, Any]]:
    if not isinstance(input_items, list):
        return []

    messages: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role")
        if role in {"system", "user", "assistant"}:
            message: dict[str, Any] = {
                "role": role,
                "content": content_parts_to_text(item.get("content")),
            }
            if role == "assistant" and item.get("tool_calls"):
                message["tool_calls"] = item["tool_calls"]
            messages.append(message)
            continue

        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": str(item.get("output", "")),
                }
            )
            continue

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or "call_0"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
    return messages


def responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function: dict[str, Any] = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters")
            or {"type": "object", "properties": {}},
        }
        if "strict" in tool:
            function["strict"] = tool["strict"]
        converted.append({"type": "function", "function": function})
    return converted


def normalize_responses_request(
    body: dict[str, Any], *, disable_thinking: bool = False
) -> dict[str, Any]:
    chat_body: dict[str, Any] = {
        "model": body.get("model"),
        "messages": responses_input_to_messages(body.get("input")),
        "stream": bool(body.get("stream", True)),
    }

    for source_key, target_key in (
        ("max_output_tokens", "max_tokens"),
        ("max_completion_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("parallel_tool_calls", "parallel_tool_calls"),
        ("tool_choice", "tool_choice"),
        ("stop", "stop"),
    ):
        if source_key in body:
            chat_body[target_key] = body[source_key]

    tools = responses_tools_to_chat_tools(body.get("tools"))
    if tools:
        chat_body["tools"] = tools

    return normalize_chat_completions_request(
        chat_body, disable_thinking=disable_thinking
    )


def prepare_upstream_request(
    path: str, body: bytes | None, *, disable_thinking: bool = False
) -> tuple[str, bytes | None, str]:
    if not body:
        return path, body, RESPONSE_MODE_PASSTHROUGH

    route = urllib.parse.urlsplit(path).path.rstrip("/")
    if route in {"/v1/chat/completions", "/chat/completions"}:
        return (
            path,
            normalize_request(path, body, disable_thinking=disable_thinking),
            RESPONSE_MODE_PASSTHROUGH,
        )

    if route in {"/v1/responses", "/responses"}:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return path, body, RESPONSE_MODE_PASSTHROUGH
        if not isinstance(payload, dict):
            return path, body, RESPONSE_MODE_PASSTHROUGH
        normalized = normalize_responses_request(
            payload, disable_thinking=disable_thinking
        )
        rewritten = json.dumps(
            normalized, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return "/v1/chat/completions", rewritten, RESPONSE_MODE_OPENAI_RESPONSES

    return path, body, RESPONSE_MODE_PASSTHROUGH


def normalize_stream_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Alias GLM reasoning_content chunks to a standard reasoning field."""
    changed = False
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload, changed

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content and "reasoning" not in delta:
            delta["reasoning"] = reasoning_content
            changed = True

    return payload, changed


def normalize_sse_line(line: bytes) -> bytes:
    if not line.startswith(b"data:"):
        return line

    payload_bytes = line[len(b"data:") :].strip()
    if not payload_bytes or payload_bytes == b"[DONE]":
        return line

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return line

    if not isinstance(payload, dict):
        return line

    normalized, changed = normalize_stream_payload(payload)
    if not changed:
        return line

    return (
        b"data: "
        + json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )


def sse_data(payload: dict[str, Any]) -> bytes:
    return (
        b"data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )


class ResponsesStreamAdapter:
    def __init__(self) -> None:
        self.response_id = f"resp_{int(time.time() * 1000)}"
        self.created_at = int(time.time())
        self.model = ""
        self.output: list[dict[str, Any]] = []
        self.started = False
        self.message_index: int | None = None
        self.message_content_started = False
        self.reasoning_index: int | None = None
        self.reasoning_content_started = False
        self.tool_indexes: dict[int, int] = {}

    def convert_line(self, line: bytes) -> bytes:
        if not line.startswith(b"data:"):
            return line

        payload_bytes = line[len(b"data:") :].strip()
        if not payload_bytes:
            return b""
        if payload_bytes == b"[DONE]":
            return self._complete_events() + b"data: [DONE]\n"

        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return line

        events: list[bytes] = []
        if not self.started:
            self.model = str(payload.get("model") or self.model)
            self.response_id = str(payload.get("id") or self.response_id)
            self.created_at = int(payload.get("created") or self.created_at)
            self.started = True
            events.append(
                sse_data(
                    {
                        "type": "response.created",
                        "response": self._response(status="in_progress"),
                    }
                )
            )

        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue

            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                events.extend(self._reasoning_events(str(reasoning)))

            content = delta.get("content")
            if content:
                events.extend(self._message_events(str(content)))

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        events.extend(self._tool_call_events(tool_call))

        return b"\n".join(events) + (b"\n" if events else b"")

    def _response(self, *, status: str) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": self.output,
        }

    def _message_events(self, text: str) -> list[bytes]:
        events: list[bytes] = []
        if self.message_index is None:
            self.message_index = len(self.output)
            item = {
                "id": f"msg_{self.message_index}",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            self.output.append(item)
            events.append(
                sse_data(
                    {
                        "type": "response.output_item.added",
                        "response_id": self.response_id,
                        "output_index": self.message_index,
                        "item": item,
                    }
                )
            )

        item = self.output[self.message_index]
        if not self.message_content_started:
            self.message_content_started = True
            part = {"type": "output_text", "text": "", "annotations": []}
            item["content"].append(part)
            events.append(
                sse_data(
                    {
                        "type": "response.content_part.added",
                        "response_id": self.response_id,
                        "output_index": self.message_index,
                        "content_index": 0,
                        "part": part,
                    }
                )
            )

        item["content"][0]["text"] += text
        events.append(
            sse_data(
                {
                    "type": "response.output_text.delta",
                    "response_id": self.response_id,
                    "output_index": self.message_index,
                    "content_index": 0,
                    "delta": text,
                }
            )
        )
        return events

    def _reasoning_events(self, text: str) -> list[bytes]:
        events: list[bytes] = []
        if self.reasoning_index is None:
            self.reasoning_index = len(self.output)
            item = {
                "id": f"rs_{self.reasoning_index}",
                "type": "reasoning",
                "status": "in_progress",
                "content": [],
            }
            self.output.append(item)
            events.append(
                sse_data(
                    {
                        "type": "response.output_item.added",
                        "response_id": self.response_id,
                        "output_index": self.reasoning_index,
                        "item": item,
                    }
                )
            )

        item = self.output[self.reasoning_index]
        if not self.reasoning_content_started:
            self.reasoning_content_started = True
            part = {"type": "reasoning_text", "text": ""}
            item["content"].append(part)
            events.append(
                sse_data(
                    {
                        "type": "response.content_part.added",
                        "response_id": self.response_id,
                        "output_index": self.reasoning_index,
                        "content_index": 0,
                        "part": part,
                    }
                )
            )

        item["content"][0]["text"] += text
        events.append(
            sse_data(
                {
                    "type": "response.reasoning_text.delta",
                    "response_id": self.response_id,
                    "output_index": self.reasoning_index,
                    "content_index": 0,
                    "delta": text,
                }
            )
        )
        return events

    def _tool_call_events(self, tool_call: dict[str, Any]) -> list[bytes]:
        events: list[bytes] = []
        index = int(tool_call.get("index") or 0)
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            function = {}

        if index not in self.tool_indexes:
            output_index = len(self.output)
            self.tool_indexes[index] = output_index
            call_id = tool_call.get("id") or f"call_{index}"
            item = {
                "id": call_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": function.get("name") or "",
                "arguments": "",
            }
            self.output.append(item)
            events.append(
                sse_data(
                    {
                        "type": "response.output_item.added",
                        "response_id": self.response_id,
                        "output_index": output_index,
                        "item": item,
                    }
                )
            )

        output_index = self.tool_indexes[index]
        item = self.output[output_index]
        if function.get("name"):
            item["name"] = function["name"]
        arguments_delta = function.get("arguments")
        if arguments_delta:
            item["arguments"] += str(arguments_delta)
            events.append(
                sse_data(
                    {
                        "type": "response.function_call_arguments.delta",
                        "response_id": self.response_id,
                        "output_index": output_index,
                        "delta": str(arguments_delta),
                    }
                )
            )
        return events

    def _complete_events(self) -> bytes:
        for item in self.output:
            item["status"] = "completed"
        return sse_data(
            {
                "type": "response.completed",
                "response": self._response(status="completed"),
            }
        )


def build_target_url(target_base: str, path: str) -> str:
    target = urllib.parse.urlsplit(target_base.rstrip("/"))
    request = urllib.parse.urlsplit(path)
    base_path = target.path.rstrip("/")
    request_path = request.path
    if base_path and request_path.startswith(base_path + "/"):
        final_path = request_path
    else:
        final_path = base_path + request_path
    return urllib.parse.urlunsplit(
        (target.scheme, target.netloc, final_path, request.query, "")
    )


class CompatProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    target_base = ""
    log_path = ""
    disable_thinking = False

    def do_GET(self) -> None:
        self._proxy(None)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        target_path, rewritten, response_mode = prepare_upstream_request(
            self.path, body, disable_thinking=self.disable_thinking
        )
        self._write_log(body, rewritten)
        self._proxy(rewritten, target_path=target_path, response_mode=response_mode)

    def _proxy(
        self,
        body: bytes | None,
        *,
        target_path: str | None = None,
        response_mode: str = RESPONSE_MODE_PASSTHROUGH,
    ) -> None:
        target_url = build_target_url(self.target_base, target_path or self.path)
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers["Host"] = urllib.parse.urlsplit(self.target_base).netloc
        if body is not None:
            headers["Content-Type"] = headers.get("Content-Type", "application/json")

        request = urllib.request.Request(
            target_url,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            response = urllib.request.urlopen(request, timeout=3600)
            self._stream_response(
                response.status, response.headers, response, response_mode=response_mode
            )
        except urllib.error.HTTPError as exc:
            self._stream_response(
                exc.code, exc.headers, exc, response_mode=RESPONSE_MODE_PASSTHROUGH
            )
        except Exception as exc:  # pragma: no cover - defensive network fallback
            data = f"{type(exc).__name__}: {exc}".encode("utf-8", "replace")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _write_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
        self.wfile.write(chunk)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _stream_response(
        self,
        status: int,
        headers: Any,
        response: Any,
        *,
        response_mode: str = RESPONSE_MODE_PASSTHROUGH,
    ) -> None:
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        content_type = headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            responses_adapter = (
                ResponsesStreamAdapter()
                if response_mode == RESPONSE_MODE_OPENAI_RESPONSES
                else None
            )
            while True:
                line = response.readline()
                if not line:
                    break
                if responses_adapter is not None:
                    self._write_chunk(responses_adapter.convert_line(line))
                else:
                    self._write_chunk(normalize_sse_line(line))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            self._write_chunk(chunk)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _write_log(self, original: bytes, rewritten: bytes | None) -> None:
        if not self.log_path:
            return
        record: dict[str, Any] = {
            "time": time.time(),
            "path": self.path,
            "rewritten": original != (rewritten or b""),
        }
        try:
            record["original"] = json.loads(original.decode("utf-8"))
        except Exception:
            record["original"] = original.decode("utf-8", "replace")
        try:
            record["normalized"] = json.loads((rewritten or b"").decode("utf-8"))
        except Exception:
            record["normalized"] = (rewritten or b"").decode("utf-8", "replace")
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format % args)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bind",
        default=os.environ.get("OPENCODE_COMPAT_PROXY_BIND", "172.16.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("OPENCODE_COMPAT_PROXY_PORT", "30004")),
    )
    parser.add_argument(
        "--target",
        default=os.environ.get("OPENCODE_COMPAT_PROXY_TARGET", "http://172.16.0.1:30003"),
    )
    parser.add_argument(
        "--log-path",
        default=os.environ.get("OPENCODE_COMPAT_PROXY_LOG", ""),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=env_flag("OPENCODE_COMPAT_DISABLE_THINKING", False),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CompatProxyHandler.target_base = args.target
    CompatProxyHandler.log_path = args.log_path
    CompatProxyHandler.disable_thinking = args.disable_thinking
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.bind, args.port), CompatProxyHandler)
    print(
        f"OpenCode compat proxy listening on {args.bind}:{args.port} -> {args.target}",
        flush=True,
    )
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
