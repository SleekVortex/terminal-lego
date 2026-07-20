from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests

from generator.tracing import GenerationTraceWriter


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "claude-opus-4-6"
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 300
REQUEST_INTERVAL = 0.2

logger = logging.getLogger(__name__)


class TokenTracker:
    def __init__(self, output_file: str = "api_token_usage.json"):
        self.output_file = output_file
        self.lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_duration_sec = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.by_stage: dict[str, dict[str, Any]] = {}

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
        task_name: str | None = None,
        stage: str | None = None,
        duration_sec: float = 0.0,
        success: bool = True,
        attempt: int = 1,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        stage_name = stage or "unknown"
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        duration = float(duration_sec or 0.0)
        total_tokens = prompt_tokens + completion_tokens
        with self.lock:
            self.records.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
                "task_name": task_name,
                "stage": stage_name,
                "attempt": attempt,
                "success": success,
                "status": status,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "duration_sec": round(duration, 3),
                "error": error,
            })
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_duration_sec += duration
            self.success_count += int(success)
            self.failure_count += int(not success)

            stage_stats = self.by_stage.setdefault(stage_name, {
                "call_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "duration_sec": 0.0,
            })
            stage_stats["call_count"] += 1
            stage_stats["success_count"] += int(success)
            stage_stats["failure_count"] += int(not success)
            stage_stats["prompt_tokens"] += prompt_tokens
            stage_stats["completion_tokens"] += completion_tokens
            stage_stats["total_tokens"] += total_tokens
            stage_stats["duration_sec"] = round(stage_stats["duration_sec"] + duration, 3)
            self._save()

    def _save(self) -> None:
        data = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_duration_sec": round(self.total_duration_sec, 3),
            "call_count": len(self.records),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "by_stage": self.by_stage,
            "records": self.records[-500:],
        }
        try:
            with open(self.output_file, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_duration_sec": round(self.total_duration_sec, 3),
                "call_count": len(self.records),
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "by_stage": self.by_stage,
            }


token_tracker = TokenTracker()
_api_lock = threading.Lock()
_last_request_time = 0.0
_config = {
    "api_base": DEFAULT_API_BASE,
    "api_key": "",
    "model": DEFAULT_MODEL,
}
_trace_writer: GenerationTraceWriter | None = None


def configure_trace_writer(output_dir: str | None) -> GenerationTraceWriter | None:
    global _trace_writer
    _trace_writer = GenerationTraceWriter(Path(output_dir)) if output_dir else None
    return _trace_writer


def get_trace_writer() -> GenerationTraceWriter | None:
    return _trace_writer


def record_trace_event(task_name: str | None, event: str, **data: Any) -> None:
    if _trace_writer is not None and task_name:
        try:
            _trace_writer.record(task_name, event, **data)
        except Exception as exc:
            logger.warning("Failed to write generation trace for %s: %s", task_name, exc)


def is_truncated(content: str) -> bool:
    if not content:
        return True
    content_stripped = content.strip()
    if len(content_stripped) < 100:
        return False
    if content_stripped.startswith("{") and not content_stripped.endswith("}"):
        if content_stripped.count("{") > content_stripped.count("}"):
            return True
    if "```json" in content:
        last_json_block = content.rfind("```json")
        after_json = content[last_json_block + 7 :]
        if "```" not in after_json:
            return True
    return False


def call_llm_api(
    prompt: str,
    system_prompt: str | None = None,
    temperature: Optional[float] = None,
    check_truncation: bool = False,
    task_name: str | None = None,
    stage: str | None = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    global _last_request_time

    headers = {
        "Authorization": f"Bearer {_config['api_key']}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {"model": _config["model"], "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    for attempt in range(MAX_RETRIES):
        attempt_started = time.time()
        with _api_lock:
            elapsed = time.time() - _last_request_time
            if elapsed < REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL - elapsed)
            _last_request_time = time.time()

        try:
            response = requests.post(
                f"{_config['api_base']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(30, REQUEST_TIMEOUT),
            )
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * (attempt + 2)))
                duration_sec = time.time() - attempt_started
                token_tracker.record(
                    0,
                    0,
                    _config["model"],
                    task_name,
                    stage=stage,
                    duration_sec=duration_sec,
                    success=False,
                    attempt=attempt + 1,
                    status="rate_limited",
                    error=f"retry_after={retry_after}",
                )
                record_trace_event(
                    task_name,
                    "llm_call",
                    stage=stage or "unknown",
                    attempt=attempt + 1,
                    status="rate_limited",
                    duration_sec=round(duration_sec, 3),
                    request=payload,
                    response=None,
                    error=f"retry_after={retry_after}",
                )
                logger.warning("Rate limited, waiting %ss...", retry_after)
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            truncated = check_truncation and is_truncated(content)
            retry_truncated = truncated and attempt < MAX_RETRIES - 1
            duration_sec = time.time() - attempt_started
            status = "truncated_retry" if retry_truncated else ("truncated_final" if truncated else "ok")
            token_tracker.record(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                _config["model"],
                task_name,
                stage=stage,
                duration_sec=duration_sec,
                success=not retry_truncated,
                attempt=attempt + 1,
                status=status,
            )
            record_trace_event(
                task_name,
                "llm_call",
                stage=stage or "unknown",
                attempt=attempt + 1,
                status=status,
                duration_sec=round(duration_sec, 3),
                request=payload,
                response=result,
                error=None,
            )
            if retry_truncated:
                logger.warning("Output truncated (attempt %s), retrying...", attempt + 1)
                time.sleep(RETRY_DELAY)
                continue
            return content
        except requests.exceptions.Timeout:
            duration_sec = time.time() - attempt_started
            token_tracker.record(
                0,
                0,
                _config["model"],
                task_name,
                stage=stage,
                duration_sec=duration_sec,
                success=False,
                attempt=attempt + 1,
                status="timeout",
                error="request timeout",
            )
            record_trace_event(
                task_name,
                "llm_call",
                stage=stage or "unknown",
                attempt=attempt + 1,
                status="timeout",
                duration_sec=round(duration_sec, 3),
                request=payload,
                response=None,
                error="request timeout",
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            duration_sec = time.time() - attempt_started
            error = f"status={status_code}: {exc}"
            token_tracker.record(
                0,
                0,
                _config["model"],
                task_name,
                stage=stage,
                duration_sec=duration_sec,
                success=False,
                attempt=attempt + 1,
                status="http_error",
                error=error,
            )
            record_trace_event(
                task_name,
                "llm_call",
                stage=stage or "unknown",
                attempt=attempt + 1,
                status="http_error",
                duration_sec=round(duration_sec, 3),
                request=payload,
                response=None,
                error=error,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt if status_code == 429 else attempt + 1))
        except Exception as exc:
            duration_sec = time.time() - attempt_started
            token_tracker.record(
                0,
                0,
                _config["model"],
                task_name,
                stage=stage,
                duration_sec=duration_sec,
                success=False,
                attempt=attempt + 1,
                status="api_error",
                error=str(exc),
            )
            record_trace_event(
                task_name,
                "llm_call",
                stage=stage or "unknown",
                attempt=attempt + 1,
                status="api_error",
                duration_sec=round(duration_sec, 3),
                request=payload,
                response=None,
                error=str(exc),
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None
