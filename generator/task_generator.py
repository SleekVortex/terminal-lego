#!/usr/bin/env python3
"""
Terminal-Lego Task Generator

Converts StackOverflow questions into Terminal Bench format tasks using an LLM.
Each task includes: instruction.md, environment/, solution/solve.sh, tests/, Dockerfile.

Generation order (cascaded):
  instruction → environment → solution → difficulty → tests (3x gen+static review) → dockerfile

Usage:
    python task_generator.py --input so_data.json --output ./candidates --workers 16
"""

import ast
import json
import os
import re
import html
import hashlib
import argparse
import logging
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import threading
import requests

# ============================================================================
# Configuration (defaults, overridable via CLI/env)
# ============================================================================
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "claude-opus-4-6"

MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 300
REQUEST_INTERVAL = 0.2

TEMP_INSTRUCTION = 0.7
TEMP_ENVIRONMENT = 0.3
TEMP_SOLUTION = 0.3
TEMP_TESTS = 0.2
TEMP_DOCKERFILE = 0.3

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('task_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value


progress_counter = Counter()
error_counter = Counter()
skip_counter = Counter()


class TokenTracker:
    def __init__(self, output_file: str = "api_token_usage.json"):
        self.output_file = output_file
        self.lock = threading.Lock()
        self.records = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_duration_sec = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.by_stage: Dict[str, Dict[str, Any]] = {}

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = None,
        task_name: str = None,
        stage: str = None,
        duration_sec: float = 0.0,
        success: bool = True,
        attempt: int = 1,
        status: str = "ok",
        error: str = None,
    ):
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
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1

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

    def _save(self):
        data = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_duration_sec": round(self.total_duration_sec, 3),
            "call_count": len(self.records),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "by_stage": self.by_stage,
            "records": self.records[-500:]
        }
        try:
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_summary(self):
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


@dataclass
class SOQuestion:
    question_id: int
    title: str
    body: str
    tags: List[str]
    score: int
    accepted_answer_id: Optional[int]
    accepted_answer_body: Optional[str]
    accepted_answer_score: Optional[int]
    link: str
    categories: List[str]
    selected_category: Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> 'SOQuestion':
        answer = data.get('accepted_answer', {})
        categories = data.get('categories', [])
        if isinstance(categories, str):
            categories = [categories]
        return cls(
            question_id=data['question_id'],
            title=data['title'],
            body=data['body'],
            tags=data.get('tags', []),
            score=data.get('score', 0),
            accepted_answer_id=data.get('accepted_answer_id'),
            accepted_answer_body=answer.get('body') if answer else None,
            accepted_answer_score=answer.get('score') if answer else None,
            link=data.get('link', ''),
            categories=categories,
            selected_category=data.get('selected_category') or data.get('category'),
        )


def clean_html(html_content: str) -> str:
    if not html_content:
        return ""

    code_placeholders: Dict[str, str] = {}

    def stash_code_block(match: re.Match) -> str:
        placeholder = f"@@TL_CODE_BLOCK_{len(code_placeholders)}@@"
        code_placeholders[placeholder] = f"```\n{html.unescape(match.group(1))}\n```"
        return placeholder

    def stash_inline_code(match: re.Match) -> str:
        placeholder = f"@@TL_INLINE_CODE_{len(code_placeholders)}@@"
        code_placeholders[placeholder] = f"`{html.unescape(match.group(1))}`"
        return placeholder

    text = re.sub(
        r'<pre[^>]*><code[^>]*>(.*?)</code></pre>',
        stash_code_block,
        html_content,
        flags=re.DOTALL,
    )
    text = re.sub(r'<code>(.*?)</code>', stash_inline_code, text, flags=re.DOTALL)
    text = html.unescape(text)
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.DOTALL)
    text = re.sub(r'<li>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<ul>|</ul>|<ol>|</ol>', '', text)
    text = re.sub(r'<p>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    for placeholder, replacement in code_placeholders.items():
        text = text.replace(placeholder, replacement)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================================
# LLM API
# ============================================================================
_api_lock = threading.Lock()
_last_request_time = 0

# Module-level config (set in main)
_config = {
    "api_base": DEFAULT_API_BASE,
    "api_key": "",
    "model": DEFAULT_MODEL,
}


def _is_truncated(content: str) -> bool:
    if not content:
        return True
    content_stripped = content.strip()
    if len(content_stripped) < 100:
        return False
    if content_stripped.startswith('{') and not content_stripped.endswith('}'):
        if content_stripped.count('{') > content_stripped.count('}'):
            return True
    if '```json' in content:
        last_json_block = content.rfind('```json')
        if last_json_block != -1:
            after_json = content[last_json_block + 7:]
            if '```' not in after_json:
                return True
    return False


def call_llm_api(
    prompt: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    check_truncation: bool = False,
    task_name: str = None,
    stage: str = None,
    max_tokens: int = 16384,
) -> Optional[str]:
    global _last_request_time

    headers = {
        "Authorization": f"Bearer {_config['api_key']}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": _config['model'],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

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
                timeout=(30, REQUEST_TIMEOUT)
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', RETRY_DELAY * (attempt + 2)))
                token_tracker.record(
                    0,
                    0,
                    _config['model'],
                    task_name,
                    stage=stage,
                    duration_sec=time.time() - attempt_started,
                    success=False,
                    attempt=attempt + 1,
                    status="rate_limited",
                    error=f"retry_after={retry_after}",
                )
                logger.warning(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']

            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            is_truncated = check_truncation and _is_truncated(content)
            will_retry_truncated = is_truncated and attempt < MAX_RETRIES - 1
            status = "ok"
            if will_retry_truncated:
                status = "truncated_retry"
            elif is_truncated:
                status = "truncated_final"
            token_tracker.record(
                prompt_tokens,
                completion_tokens,
                _config['model'],
                task_name,
                stage=stage,
                duration_sec=time.time() - attempt_started,
                success=not will_retry_truncated,
                attempt=attempt + 1,
                status=status,
            )

            if is_truncated:
                logger.warning(f"Output truncated (attempt {attempt + 1}), retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue

            return content
        except requests.exceptions.Timeout:
            token_tracker.record(
                0,
                0,
                _config['model'],
                task_name,
                stage=stage,
                duration_sec=time.time() - attempt_started,
                success=False,
                attempt=attempt + 1,
                status="timeout",
                error="request timeout",
            )
            logger.warning(f"Timeout (attempt {attempt + 1})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            token_tracker.record(
                0,
                0,
                _config['model'],
                task_name,
                stage=stage,
                duration_sec=time.time() - attempt_started,
                success=False,
                attempt=attempt + 1,
                status="http_error",
                error=f"status={status_code}: {e}",
            )
            if e.response and e.response.status_code == 429:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            token_tracker.record(
                0,
                0,
                _config['model'],
                task_name,
                stage=stage,
                duration_sec=time.time() - attempt_started,
                success=False,
                attempt=attempt + 1,
                status="api_error",
                error=str(e),
            )
            logger.warning(f"API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return None


PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompts" / "task_generator"


def _load_prompt_template(filename: str) -> str:
    return (PROMPT_TEMPLATE_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


SYSTEM_PROMPT = _load_prompt_template("system.md")
INSTRUCTION_PROMPT_TEMPLATE = _load_prompt_template("instruction.md")
ENVIRONMENT_PROMPT_TEMPLATE = _load_prompt_template("environment.md")
SOLUTION_PROMPT_TEMPLATE = _load_prompt_template("solution.md")
TEST_PROMPT_TEMPLATE = _load_prompt_template("tests.md")
DOCKERFILE_PROMPT_TEMPLATE = _load_prompt_template("dockerfile.md")


class TaskGenerator:
    def __init__(self, question: SOQuestion, output_dir: Path, index: int = None):
        self.question = question
        self.output_dir = output_dir
        if index is not None:
            self.task_name = f"task_{index:05d}"
        else:
            self.task_name = self._generate_task_name()
        self.task_dir = output_dir / self.task_name
        self.guid = hashlib.md5(f"{question.question_id}".encode()).hexdigest()[:8]

    def _generate_task_name(self) -> str:
        title = self.question.title.lower()
        title = re.sub(r'[^a-z0-9\s-]', '', title)
        title = re.sub(r'\s+', '-', title)
        title = title[:50].rstrip('-')
        return f"{title}-{self.question.question_id}"

    def generate(self) -> bool:
        try:
            instruction = self._generate_instruction()
            if not instruction:
                logger.error(f"[{self.task_name}] Failed to generate instruction")
                return False

            env_data = self._generate_environment(instruction)

            solution = self._generate_solution(instruction, env_data)
            if not solution:
                logger.error(f"[{self.task_name}] Failed to generate solution")
                return False

            difficulty = self._assess_difficulty(instruction)

            test_data = self._generate_tests(instruction, env_data, solution)
            if not test_data:
                logger.error(f"[{self.task_name}] Failed to generate valid tests")
                skip_counter.increment()
                return False

            dockerfile = self._generate_dockerfile(instruction)

            self._write_files(instruction, env_data, test_data, solution, dockerfile, difficulty)
            return True

        except Exception as e:
            logger.exception(f"[{self.task_name}] Exception: {e}")
            return False

    def _generate_instruction(self) -> Optional[str]:
        prompt = INSTRUCTION_PROMPT_TEMPLATE.format(
            title=self.question.title,
            tags=', '.join(self.question.tags),
            body=clean_html(self.question.body)
        )
        response = call_llm_api(
            prompt,
            SYSTEM_PROMPT,
            temperature=TEMP_INSTRUCTION,
            check_truncation=True,
            task_name=self.task_name,
            stage="instruction",
        )
        if not response:
            return None
        response_stripped = response.strip()
        match = re.search(r'^```markdown\n(.*?)```$', response_stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
        if response_stripped.startswith('```markdown\n'):
            response_stripped = response_stripped[12:]
            if response_stripped.endswith('```'):
                response_stripped = response_stripped[:-3]
            return response_stripped.strip()
        return response_stripped

    def _generate_environment(self, instruction: str) -> Dict[str, Any]:
        prompt = ENVIRONMENT_PROMPT_TEMPLATE.format(
            instruction=instruction,
            title=self.question.title,
            tags=', '.join(self.question.tags)
        )
        response = call_llm_api(
            prompt,
            SYSTEM_PROMPT,
            temperature=TEMP_ENVIRONMENT,
            task_name=self.task_name,
            stage="environment",
        )
        if not response:
            return {"files": {}, "directories": ["task_file"]}
        match = re.search(r'```json\n?(.*?)```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"files": {}, "directories": ["task_file"]}

    def _format_env_file_list(self, env_data: Dict[str, Any]) -> str:
        lines = []
        for dir_path in env_data.get("directories", []):
            lines.append(f"  [dir]  /app/{dir_path}/")
        for file_path, content in env_data.get("files", {}).items():
            preview = content[:200] + ("..." if len(content) > 200 else "")
            lines.append(f"  [file] /app/{file_path}")
            lines.append(f"         Content: {preview}")
        return "\n".join(lines) if lines else "  (no environment files)"

    def _generate_solution(self, instruction: str, env_data: Dict[str, Any]) -> Optional[str]:
        answer = clean_html(self.question.accepted_answer_body) if self.question.accepted_answer_body else ""
        prompt = SOLUTION_PROMPT_TEMPLATE.format(
            instruction=instruction,
            answer=answer,
            tags=', '.join(self.question.tags),
            env_file_list=self._format_env_file_list(env_data),
        )
        response = call_llm_api(
            prompt,
            SYSTEM_PROMPT,
            temperature=TEMP_SOLUTION,
            task_name=self.task_name,
            stage="solution",
        )
        if not response:
            return None
        match = re.search(r'```bash\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```sh\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def _generate_tests(self, instruction: str, env_data: Dict[str, Any], solution: str) -> Optional[Dict[str, str]]:
        env_file_list = self._format_env_file_list(env_data)
        candidates = []

        for attempt in range(3):
            prompt = TEST_PROMPT_TEMPLATE.format(
                instruction=instruction,
                env_file_list=env_file_list,
                solution=solution,
                tags=', '.join(self.question.tags),
            )
            response = call_llm_api(
                prompt,
                SYSTEM_PROMPT,
                temperature=TEMP_TESTS,
                task_name=self.task_name,
                stage="tests",
            )
            if not response:
                continue

            test_data = None
            match = re.search(r'```json\n?(.*?)```', response, re.DOTALL)
            if match:
                try:
                    test_data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            if not test_data:
                try:
                    test_data = json.loads(response)
                except json.JSONDecodeError:
                    continue

            test_py = test_data.get("test_outputs_py", "")
            if not test_py:
                continue
            try:
                ast.parse(test_py)
            except SyntaxError as e:
                logger.warning(f"[{self.task_name}] ast.parse failed (round {attempt + 1}): {e.msg}")
                continue

            review = self._review_tests(instruction, env_file_list, solution, test_py)
            if review.get("pass"):
                logger.info(f"[{self.task_name}] test review passed (round {attempt + 1})")
                return test_data
            else:
                issues = review.get("issues", [])
                logger.warning(f"[{self.task_name}] test review rejected (round {attempt + 1}): {issues}")
                candidates.append((test_data, issues))

        if candidates:
            return candidates[0][0]
        return None

    def _review_tests(self, instruction: str, env_file_list: str, solution: str, test_code: str) -> dict:
        issues = []
        try:
            tree = ast.parse(test_code)
        except SyntaxError as exc:
            return {"pass": False, "issues": [f"Python syntax error: {exc.msg}"]}

        forbidden_runners = {
            "run",
            "call",
            "check_call",
            "check_output",
            "Popen",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                owner_name = None
                if isinstance(func, ast.Attribute):
                    func_name = func.attr
                    if isinstance(func.value, ast.Name):
                        owner_name = func.value.id
                elif isinstance(func, ast.Name):
                    func_name = func.id
                call_text = ast.get_source_segment(test_code, node) or ""
                if "solve.sh" in call_text and (
                    (owner_name == "subprocess" and func_name in forbidden_runners)
                    or (owner_name == "os" and func_name in {"system", "popen"})
                    or func_name in {"system", "popen"}
                ):
                    issues.append("test_outputs.py must not execute solve.sh")

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_names = []
                if isinstance(node, ast.Import):
                    module_names = [alias.name.split(".", 1)[0] for alias in node.names]
                elif node.module:
                    module_names = [node.module.split(".", 1)[0]]
                for module_name in module_names:
                    if module_name == "pytest":
                        continue
                    if module_name not in sys.stdlib_module_names:
                        issues.append(f"non-stdlib import in test_outputs.py: {module_name}")

        return {"pass": not issues, "issues": issues}

    def _generate_dockerfile(self, instruction: str) -> str:
        prompt = DOCKERFILE_PROMPT_TEMPLATE.format(
            instruction=instruction,
            tags=', '.join(self.question.tags),
        )
        response = call_llm_api(
            prompt,
            SYSTEM_PROMPT,
            temperature=TEMP_DOCKERFILE,
            task_name=self.task_name,
            stage="dockerfile",
        )
        if not response:
            return self._default_dockerfile()
        match = re.search(r'```dockerfile\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return self._default_dockerfile()

    def _assess_difficulty(self, instruction: str) -> str:
        body = clean_html(self.question.body)
        answer = clean_html(self.question.accepted_answer_body or "")
        body_len = len(body)
        context_len = body_len + len(answer)
        code_blocks = body.count("```") // 2

        if body_len <= 1200 and context_len <= 3000 and code_blocks <= 1:
            return "easy"
        if body_len >= 5000 or context_len >= 10000 or code_blocks >= 4:
            return "hard"
        return "medium"

    def _default_dockerfile(self) -> str:
        return """FROM ubuntu:22.04
WORKDIR /app

RUN apt-get update && apt-get install -y \\
    bash coreutils findutils grep sed gawk curl wget git \\
    && rm -rf /var/lib/apt/lists/*

COPY ./task_file /app/task_file
"""

    def _toml_quote(self, value: Any) -> str:
        return json.dumps(str(value), ensure_ascii=False)

    def _generate_task_toml(self, difficulty: str = "medium") -> str:
        tags = self.question.tags
        category_mapping = {
            'linux': 'system-administration', 'bash': 'shell-scripting',
            'python': 'programming', 'git': 'version-control',
            'docker': 'containerization', 'nginx': 'web-server',
            'ssh': 'networking', 'networking': 'networking',
            'database': 'database', 'sql': 'database',
        }
        category = self.question.selected_category
        if not category and self.question.categories:
            category = self.question.categories[0]
        if not category:
            category = 'general'
            for tag in tags:
                if tag.lower() in category_mapping:
                    category = category_mapping[tag.lower()]
                    break

        tags_str = ', '.join(self._toml_quote(t) for t in tags[:5])
        categories_str = ', '.join(self._toml_quote(c) for c in self.question.categories)
        return f'''version = "1.0"

[metadata]
author_name = "StackOverflow Community"
author_email = "community@stackoverflow.com"
difficulty = {self._toml_quote(difficulty)}
category = {self._toml_quote(category)}
tags = [{tags_str}]
categories = [{categories_str}]
source_url = {self._toml_quote(self.question.link)}
source_score = {self.question.score}

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 600.0

[environment]
build_timeout_sec = 120.0
cpus = 1
memory = "1G"
storage = "5G"
'''

    def _write_files(self, instruction: str, env_data: Dict, test_data: Dict, solution: str, dockerfile: str, difficulty: str = "medium"):
        self.task_dir.mkdir(parents=True, exist_ok=True)
        (self.task_dir / "environment").mkdir(exist_ok=True)
        (self.task_dir / "tests").mkdir(exist_ok=True)
        (self.task_dir / "solution").mkdir(exist_ok=True)

        (self.task_dir / "instruction.md").write_text(instruction, encoding='utf-8')
        (self.task_dir / "task.toml").write_text(self._generate_task_toml(difficulty), encoding='utf-8')
        (self.task_dir / "environment" / "Dockerfile").write_text(dockerfile, encoding='utf-8')

        for dir_path in env_data.get("directories", []):
            (self.task_dir / "environment" / dir_path).mkdir(parents=True, exist_ok=True)
        for file_path, content in env_data.get("files", {}).items():
            full_path = self.task_dir / "environment" / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')

        (self.task_dir / "tests" / "__init__.py").write_text("", encoding='utf-8')
        (self.task_dir / "tests" / "test.sh").write_text(test_data.get("test_sh", ""), encoding='utf-8')
        (self.task_dir / "tests" / "test_outputs.py").write_text(test_data.get("test_outputs_py", ""), encoding='utf-8')
        (self.task_dir / "solution" / "solve.sh").write_text(solution, encoding='utf-8')

        logger.info(f"[{self.task_name}] Written to: {self.task_dir}")


# ============================================================================
# Main
# ============================================================================
def process_question(args: tuple) -> bool:
    question_data, output_dir, total_count, index = args
    try:
        question = SOQuestion.from_dict(question_data)
        generator = TaskGenerator(question, output_dir, index=index)
        success = generator.generate()
        current = progress_counter.increment()
        if success:
            logger.info(f"Progress: {current}/{total_count} - OK: {generator.task_name}")
        else:
            error_counter.increment()
            logger.error(f"Progress: {current}/{total_count} - FAIL: {generator.task_name}")
        return success
    except Exception as e:
        error_counter.increment()
        logger.exception(f"Exception processing question: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Terminal-Lego Task Generator')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file from scraper')
    parser.add_argument('--output', '-o', default='./candidates', help='Output directory')
    parser.add_argument('--workers', '-w', type=int, default=16, help='Parallel worker threads')
    parser.add_argument('--limit', '-l', type=int, default=None, help='Limit questions to process')
    parser.add_argument('--start', '-s', type=int, default=0, help='Start index')
    parser.add_argument('--api-base', type=str, default=None, help='API base URL')
    parser.add_argument('--api-key', type=str, default=None, help='API key')
    parser.add_argument('--model', type=str, default=None, help='Model name')

    args = parser.parse_args()

    _config['api_base'] = args.api_base or os.environ.get('OPENAI_API_BASE', DEFAULT_API_BASE)
    _config['api_key'] = args.api_key or os.environ.get('OPENAI_API_KEY', '')
    _config['model'] = args.model or os.environ.get('MODEL_NAME', DEFAULT_MODEL)

    if not _config['api_key']:
        logger.error("No API key provided. Use --api-key or set OPENAI_API_KEY env var.")
        sys.exit(1)

    logger.info(f"Model: {_config['model']}")
    logger.info(f"API base: {_config['api_base']}")

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = [q for q in data['questions'] if q.get('accepted_answer')]
    logger.info(f"Found {len(questions)} questions with accepted answers (of {len(data['questions'])} total)")

    questions = questions[args.start:]
    if args.limit:
        questions = questions[:args.limit]

    logger.info(f"Processing {len(questions)} questions (start={args.start})")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_count = len(questions)
    task_args = [(q, output_dir, total_count, args.start + i) for i, q in enumerate(questions)]

    start_time = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_question, arg) for arg in task_args]
        for future in as_completed(futures):
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.exception(f"Future exception: {e}")

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"Done! Time: {elapsed:.1f}s")
    logger.info(f"Success: {success_count}/{total_count}")
    logger.info(f"Failed: {error_counter.value}, Skipped: {skip_counter.value}")
    logger.info(f"Token usage: {token_tracker.get_summary()}")
    logger.info("=" * 60)

    summary = {
        "total_processed": total_count,
        "success_count": success_count,
        "error_count": error_counter.value,
        "skip_count": skip_counter.value,
        "elapsed_seconds": elapsed,
        "model": _config['model'],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "token_usage": token_tracker.get_summary(),
    }
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
