#!/usr/bin/env python3
"""
Terminal-Lego Task Generator

Converts StackOverflow questions into Terminal Bench format tasks using an LLM.
Each task includes: instruction.md, environment/, solution/solve.sh, tests/, Dockerfile.

Generation order (cascaded):
  instruction → environment → solution → difficulty → tests (3x gen+review) → dockerfile

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

    def record(self, prompt_tokens: int, completion_tokens: int, model: str = None, task_name: str = None):
        with self.lock:
            self.records.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
                "task_name": task_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self._save()

    def _save(self):
        data = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "call_count": len(self.records),
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
                "call_count": len(self.records)
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
    text = html.unescape(html_content)
    text = re.sub(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', r'```\n\1\n```', text, flags=re.DOTALL)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.DOTALL)
    text = re.sub(r'<li>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<ul>|</ul>|<ol>|</ol>', '', text)
    text = re.sub(r'<p>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
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
                logger.warning(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']

            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            if prompt_tokens > 0 or completion_tokens > 0:
                token_tracker.record(prompt_tokens, completion_tokens, _config['model'], task_name)

            if check_truncation and _is_truncated(content):
                logger.warning(f"Output truncated (attempt {attempt + 1}), retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue

            return content
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout (attempt {attempt + 1})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            logger.warning(f"API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return None


SYSTEM_PROMPT = """You are a professional technical task designer responsible for converting StackOverflow questions into Terminal Bench format programming tasks.
You need to generate clear, executable, and testable tasks. All outputs should be actual runnable code and scripts.
Please ensure the generated content is in correct format and can be directly saved as files."""

INSTRUCTION_PROMPT_TEMPLATE = """Please convert the following StackOverflow question into a Terminal Bench task instruction.md file.

**Original Question:**
Title: {title}
Tags: {tags}
Content:
{body}

**Requirements:**
1. Write a clear task description in Markdown format
2. The task should be completed in a Linux terminal environment
3. Specify clear working directory paths (use /app/task_file/ as root directory)
4. If input files are needed, specify file location (e.g., /app/task_file/input/)
5. If output files are needed, specify output location (e.g., /app/task_file/output/)
6. Provide specific success criteria

Please output the instruction.md content directly in markdown format. Do NOT wrap in code blocks."""

ENVIRONMENT_PROMPT_TEMPLATE = """Based on the following Terminal Bench task, analyze and generate the required environment files.

**Task instruction:**
{instruction}

**Original question info:**
Title: {title}
Tags: {tags}

**Requirements:**
1. Analyze what preset files the task needs (e.g., input data, config files)
2. Generate reasonable test data
3. File paths relative to environment/ directory
4. If needed, create subdirectory structure like task_file/input/

**Output format (JSON):**
```json
{{
    "files": {{
        "relative/path/filename": "file content",
        "task_file/input/example.txt": "example content..."
    }},
    "directories": ["task_file", "task_file/input", "task_file/output"]
}}
```

Please output only JSON, wrapped with ```json```. If no files are needed, return empty files and directories."""

SOLUTION_PROMPT_TEMPLATE = """Based on the following Terminal Bench task and StackOverflow answer, generate solution/solve.sh file.

**Task instruction:**
{instruction}

**StackOverflow answer:**
{answer}

**Tags:** {tags}

**Environment files available in the container:**
{env_file_list}

**Requirements:**
1. Generate an executable bash script
2. The script should complete the task requirements
3. Include necessary comments
4. Handle possible error cases
5. Ensure output meets task requirements
6. The script runs inside the container at WORKDIR /app

**Output format:**
```bash
#!/bin/bash
# Your solution code...
```

Please output only bash script content, wrapped with ```bash```."""

TEST_PROMPT_TEMPLATE = """Based on the following Terminal Bench task, its environment, and its reference solution, generate test code.

**Task instruction:**
{instruction}

**Environment files in the container:**
{env_file_list}

**Reference solution (solve.sh) that will be executed:**
```bash
{solution}
```

**Tags:** {tags}

**Generate two files:**

1. **test.sh** - Test runner script, format:
```bash
#!/bin/bash

apt-get update && apt-get install -y curl

curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    exit 1
fi

uvx \\
  -p 3.13 \\
  -w pytest==8.4.1 \\
  -w pytest-json-ctrf==0.3.5 \\
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
```

2. **test_outputs.py** - pytest test file that verifies the state AFTER solve.sh has already been executed:

**CRITICAL RULES for test_outputs.py:**
- solve.sh has ALREADY been executed before test_outputs.py runs. Do NOT call solve.sh again.
- Tests should ONLY check the resulting state: output files, file contents, directory structures, etc.
- Use os.path.exists(), open().read(), subprocess.run() for verification commands — but NEVER run solve.sh.

**ROBUSTNESS RULES:**
- When checking output files that list paths, use `line.endswith("filename")` or `os.path.basename(line)`.
- When checking file contents, use `in` operator or regex, NOT exact equality.
- When counting lines, allow for trailing newlines: `len([l for l in content.strip().splitlines() if l.strip()])`.
- Prefer checking that files/directories EXIST on disk over parsing output text.

**Output format (JSON):**
```json
{{
    "test_sh": "#!/bin/bash\\n...",
    "test_outputs_py": "import pytest\\n..."
}}
```

Please output only JSON, wrapped with ```json```. Ensure Python code syntax is correct."""

DOCKERFILE_PROMPT_TEMPLATE = """Based on the following Terminal Bench task, generate a Dockerfile for the environment.

**Task instruction:**
{instruction}

**Tags:** {tags}

**Requirements:**
1. Choose an appropriate base image:
   - Python tasks: `python:3.13-slim-bookworm`
   - Node.js tasks: `node:20-slim`
   - Java tasks: `openjdk:17-slim`
   - Go tasks: `golang:1.21-bookworm`
   - General Linux/shell tasks: `ubuntu:22.04`
2. Install necessary packages
3. Set WORKDIR to /app
4. COPY ./task_file /app/task_file

**Output format:**
```dockerfile
FROM <base_image>
WORKDIR /app
RUN apt-get update && apt-get install -y <packages> && rm -rf /var/lib/apt/lists/*
COPY ./task_file /app/task_file
```

Please output only Dockerfile content, wrapped with ```dockerfile```."""

DIFFICULTY_PROMPT_TEMPLATE = """Assess the difficulty of the following terminal/programming task for an experienced software engineer.

**Task instruction:**
{instruction}

**Tags:** {tags}

**Difficulty levels:**
- **easy**: Single-command or simple script. < 5 minutes.
- **medium**: Multiple tools or multi-step script. 5-15 minutes.
- **hard**: Deep domain knowledge or complex logic. > 15 minutes.

Respond with ONLY one word: easy, medium, or hard."""

TEST_REVIEW_PROMPT_TEMPLATE = """Review this pytest test file for a Terminal Bench task. The test runs AFTER solve.sh has already been executed.

**Task instruction:**
{instruction}

**Environment files:**
{env_file_list}

**solve.sh (already executed):**
```bash
{solution}
```

**test_outputs.py to review:**
```python
{test_code}
```

**Check for:**
1. Does the test try to run solve.sh via subprocess? (FORBIDDEN)
2. Does the test check for exact path strings that may differ?
3. Are there import errors or missing modules (only stdlib + pytest available)?
4. Do assertions match actual behavior of solve.sh?

**Respond with JSON:**
```json
{{
    "pass": true/false,
    "issues": ["issue1", ...]
}}
```"""


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
        response = call_llm_api(prompt, SYSTEM_PROMPT, temperature=TEMP_INSTRUCTION, check_truncation=True, task_name=self.task_name)
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
        response = call_llm_api(prompt, SYSTEM_PROMPT, temperature=TEMP_ENVIRONMENT, task_name=self.task_name)
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
        response = call_llm_api(prompt, SYSTEM_PROMPT, temperature=TEMP_SOLUTION, task_name=self.task_name)
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
            response = call_llm_api(prompt, SYSTEM_PROMPT, temperature=TEMP_TESTS, task_name=self.task_name)
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
                return test_data
            else:
                candidates.append((test_data, review.get("issues", [])))

        if candidates:
            return candidates[0][0]
        return None

    def _review_tests(self, instruction: str, env_file_list: str, solution: str, test_code: str) -> dict:
        prompt = TEST_REVIEW_PROMPT_TEMPLATE.format(
            instruction=instruction,
            env_file_list=env_file_list,
            solution=solution,
            test_code=test_code
        )
        response = call_llm_api(prompt, SYSTEM_PROMPT, temperature=0.1, task_name=self.task_name, max_tokens=2048)
        if not response:
            return {"pass": True, "issues": []}
        match = re.search(r'```json\n?(.*?)```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            if '"pass": true' in response.lower():
                return {"pass": True, "issues": []}
            return {"pass": False, "issues": ["Failed to parse review response"]}

    def _generate_dockerfile(self, instruction: str) -> str:
        prompt = DOCKERFILE_PROMPT_TEMPLATE.format(
            instruction=instruction,
            tags=', '.join(self.question.tags),
        )
        response = call_llm_api(prompt, SYSTEM_PROMPT, temperature=TEMP_DOCKERFILE, task_name=self.task_name)
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
        prompt = DIFFICULTY_PROMPT_TEMPLATE.format(
            instruction=instruction,
            tags=', '.join(self.question.tags)
        )
        response = call_llm_api(prompt, SYSTEM_PROMPT, temperature=0.1, task_name=self.task_name, max_tokens=16)
        if response:
            word = response.strip().lower().rstrip('.')
            if word in ("easy", "medium", "hard"):
                return word
            for level in ("easy", "medium", "hard"):
                if level in word:
                    return level
        if self.question.score > 500:
            return "easy"
        elif self.question.score > 100:
            return "medium"
        return "hard"

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
