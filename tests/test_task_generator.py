from __future__ import annotations

import json
import stat
import ast
import hashlib
from pathlib import Path

import pytest

from generator import llm_client
from generator import task_generator as cli
from generator.contracts import SOQuestion
from generator.agents.task_generation import (
    InstructionRewriteAgent,
    extract_absolute_paths,
)
from generator.instruction_style import (
    select_instruction_style,
    validate_lossy_ratio,
)
from generator.llm_client import TokenTracker, call_llm_api, is_truncated
from generator.tracing import GenerationTraceWriter, TRACE_SCHEMA_VERSION
from generator.task_builder import TaskGenerator
from generator.prompt_loader import (
    DOCKERFILE_PROMPT_TEMPLATE,
    ENVIRONMENT_PROMPT_TEMPLATE,
    INSTRUCTION_PROMPT_TEMPLATE,
    INSTRUCTION_REWRITE_CONCISE_PROMPT_TEMPLATE,
    INSTRUCTION_REWRITE_CONCISE_SYSTEM_PROMPT,
    INSTRUCTION_REWRITE_LOSSY_PROMPT_TEMPLATE,
    INSTRUCTION_REWRITE_LOSSY_SYSTEM_PROMPT,
    PROMPT_TEMPLATE_DIR,
    SOLUTION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    TEST_PROMPT_TEMPLATE,
)
from generator.resume import build_task_args, collect_existing_question_ids
from generator.reviewers.dockerfile_review import ensure_verifier_deps, review_dockerfile
from generator.reviewers.test_review import STATIC_TEST_SH, parse_test_outputs_py
from generator.text_utils import clean_html


def make_question(**overrides) -> SOQuestion:
    data = {
        "question_id": 123,
        "title": "Parse JSON with Bash?",
        "body": "<p>body</p>",
        "tags": ["bash", "json"],
        "score": 42,
        "accepted_answer_id": 456,
        "accepted_answer": {"answer_id": 456, "body": "<p>answer</p>", "score": 99},
        "link": "https://stackoverflow.com/questions/123",
        "categories": ["system-administration", "data-processing"],
        "selected_category": "system-administration",
    }
    data.update(overrides)
    return SOQuestion.from_dict(data)


def test_so_question_from_dict_normalizes_categories_and_answer() -> None:
    question = make_question(categories="security", selected_category=None, category="security")

    assert question.question_id == 123
    assert question.categories == ["security"]
    assert question.selected_category == "security"
    assert question.accepted_answer_body == "<p>answer</p>"
    assert question.accepted_answer_score == 99


def test_clean_html_preserves_code_links_lists_and_emphasis() -> None:
    html = (
        '<p>Hello <strong>world</strong></p>'
        '<pre><code>print("&lt;x&gt;")</code></pre>'
        '<p>Use <a href="https://example.com">docs</a> and <code>jq</code></p>'
        '<ul><li>one</li><li><em>two</em></li></ul>'
    )

    cleaned = clean_html(html)

    assert "**world**" in cleaned
    assert '```\nprint("<x>")\n```' in cleaned
    assert "[docs](https://example.com)" in cleaned
    assert "`jq`" in cleaned
    assert "- one" in cleaned
    assert "- *two*" in cleaned


def test_truncation_detection_for_json_and_fenced_blocks() -> None:
    assert is_truncated("{" + '"x": ' + '"y"' * 80)
    assert is_truncated("prefix ```json\n" + '{"x": 1}' * 20)
    assert not is_truncated("short")
    assert not is_truncated('```json\n{"x": 1}\n```')


def test_call_llm_api_omits_optional_generation_params_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = []

    class DummyTracker:
        def record(self, *args, **kwargs):
            pass

    class DummyResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "reason",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def fake_post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return DummyResponse()

    monkeypatch.setattr(llm_client, "token_tracker", DummyTracker())
    monkeypatch.setattr(llm_client, "_last_request_time", 0)
    monkeypatch.setitem(llm_client._config, "api_base", "http://example.test/v1")
    monkeypatch.setitem(llm_client._config, "api_key", "EMPTY")
    monkeypatch.setitem(llm_client._config, "model", "test-model")
    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    trace_dir = tmp_path / "traces"
    llm_client.configure_trace_writer(str(trace_dir))

    assert call_llm_api("prompt", system_prompt="system", task_name="task_00001", stage="instruction") == "ok"
    assert "max_tokens" not in payloads[-1]
    assert "temperature" not in payloads[-1]

    assert call_llm_api("prompt", max_tokens=123, temperature=0.4) == "ok"
    assert payloads[-1]["max_tokens"] == 123
    assert payloads[-1]["temperature"] == 0.4

    rows = [json.loads(line) for line in (trace_dir / "task_00001.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "llm_call"
    assert rows[0]["stage"] == "instruction"
    assert rows[0]["request"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert rows[0]["response"]["choices"][0]["message"]["reasoning_content"] == "reason"
    llm_client.configure_trace_writer(None)


def test_generation_trace_writer_appends_ordered_task_events(tmp_path: Path) -> None:
    writer = GenerationTraceWriter(tmp_path)
    question = make_question()

    writer.task_started("task_00001", question)
    writer.record("task_00001", "llm_call", stage="instruction", request={}, response={})
    writer.task_finished("task_00001", True)

    rows = [json.loads(line) for line in (tmp_path / "task_00001.jsonl").read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["event"] for row in rows] == ["task_started", "llm_call", "task_finished"]
    assert all(row["schema_version"] == TRACE_SCHEMA_VERSION for row in rows)
    assert all(row["run_id"] == writer.run_id for row in rows)
    assert rows[0]["source"]["question_id"] == 123
    assert rows[-1]["success"] is True


def test_task_name_generation_and_env_file_format(tmp_path: Path) -> None:
    generator = TaskGenerator(make_question(title="A/B: Parse JSON?"), tmp_path)

    assert generator.task_name == "ab-parse-json-123"
    assert generator.guid == hashlib.md5(b"123").hexdigest()[:8]

    listing = generator._format_env_file_list(
        {
            "directories": ["task_file/input"],
            "files": {"task_file/input/data.txt": "x" * 250},
        }
    )
    assert "[dir]  /app/task_file/input/" in listing
    assert "[file] /app/task_file/input/data.txt" in listing
    assert "..." in listing


def test_prompt_templates_are_loaded_from_prompt_files() -> None:
    prompt_files = {
        "system.md": SYSTEM_PROMPT,
        "instruction.md": INSTRUCTION_PROMPT_TEMPLATE,
        "instruction_rewrite_concise_system.md": INSTRUCTION_REWRITE_CONCISE_SYSTEM_PROMPT,
        "instruction_rewrite_concise.md": INSTRUCTION_REWRITE_CONCISE_PROMPT_TEMPLATE,
        "instruction_rewrite_lossy_system.md": INSTRUCTION_REWRITE_LOSSY_SYSTEM_PROMPT,
        "instruction_rewrite_lossy.md": INSTRUCTION_REWRITE_LOSSY_PROMPT_TEMPLATE,
        "environment.md": ENVIRONMENT_PROMPT_TEMPLATE,
        "solution.md": SOLUTION_PROMPT_TEMPLATE,
        "tests.md": TEST_PROMPT_TEMPLATE,
        "dockerfile.md": DOCKERFILE_PROMPT_TEMPLATE,
    }

    assert PROMPT_TEMPLATE_DIR.name == "task_generator"
    for filename, prompt in prompt_files.items():
        assert prompt == (PROMPT_TEMPLATE_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


def test_instruction_style_selection_is_deterministic() -> None:
    assert select_instruction_style("off", 123, 0.25) == "off"
    assert select_instruction_style("concise", 123, 0.25) == "concise"
    assert select_instruction_style("lossy", 123, 0.25) == "lossy"
    assert select_instruction_style("mixed", 123, 0.0) == "concise"
    assert select_instruction_style("mixed", 123, 1.0) == "lossy"
    assert select_instruction_style("mixed", 123, 0.25) == select_instruction_style(
        "mixed",
        123,
        0.25,
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_lossy_ratio(1.1)
    with pytest.raises(ValueError, match="instruction rewrite mode"):
        select_instruction_style("unknown", 123, 0.25)


def test_extract_absolute_paths_preserves_order_and_uniqueness() -> None:
    instruction = (
        "Read `/app/task_file/input/data.csv`, write '/app/task_file/output/result.json', "
        "then inspect /app/task_file/input/data.csv and keep artifacts in '/app/task_file/output/'. "
        "Do not mistake https://localhost:8080/api/test, </div>, or /* for filesystem paths. "
        "Also verify '/SHASUMS256.txt'."
    )

    assert extract_absolute_paths(instruction) == [
        "/app/task_file/input/data.csv",
        "/app/task_file/output/result.json",
        "/app/task_file/output/",
        "/SHASUMS256.txt",
    ]


def test_concise_instruction_rewrite_retries_missing_paths() -> None:
    calls = []
    responses = iter(
        [
            "I have input data under /app/task_file/input/data.csv and need it converted into a useful report without changing the source file.",
            "I have input data in /app/task_file/input/data.csv that needs conversion. Produce /app/task_file/output/result.json with the required result while preserving the source file and its values exactly.",
        ]
    )

    def fake_call(prompt: str, *args, **kwargs) -> str:
        calls.append((prompt, kwargs))
        return next(responses)

    rewritten = InstructionRewriteAgent(fake_call, "task_00001", "concise").run(
        "Read /app/task_file/input/data.csv and write /app/task_file/output/result.json."
    )

    assert rewritten is not None
    assert "/app/task_file/output/result.json" in rewritten
    assert len(calls) == 2
    assert calls[0][1]["stage"] == "instruction_rewrite_concise"
    assert "missing required paths" in calls[1][0]


def test_lossy_instruction_rewrite_allows_omitted_paths() -> None:
    response = (
        "I have a data conversion job that produces the wrong grouping and ordering. "
        "Please inspect the provided files, determine the intended behavior, and fix the implementation so the generated report is consistent."
    )
    calls = []

    def fake_call(prompt: str, *args, **kwargs) -> str:
        calls.append((prompt, kwargs))
        return response

    rewritten = InstructionRewriteAgent(fake_call, "task_00001", "lossy").run(
        "Read /app/task_file/input/data.csv and write /app/task_file/output/result.json."
    )

    assert rewritten == response
    assert "/app/" not in rewritten
    assert len(calls) == 1
    assert calls[0][1]["stage"] == "instruction_rewrite_lossy"


def test_dockerfile_prompt_is_runtime_first() -> None:
    prompt = DOCKERFILE_PROMPT_TEMPLATE

    assert "Choose the base image for the task runtime first." in prompt
    assert "Do not choose a Python image only because the verifier uses Python." in prompt
    assert "The generator will add offline verifier dependencies separately" in prompt
    assert "Prefer `python:3.12-slim-bookworm`" not in prompt
    assert "General Linux/shell tasks: `ubuntu:22.04`" in prompt
    assert "Node.js tasks: `node:20-slim`" in prompt
    assert "{env_file_list}" in prompt
    assert "{solution}" in prompt
    assert "{test_outputs_py}" in prompt


def test_token_tracker_records_stage_telemetry(tmp_path: Path) -> None:
    usage_path = tmp_path / "usage.json"
    tracker = TokenTracker(str(usage_path))

    tracker.record(
        10,
        5,
        model="model",
        task_name="task_00001",
        stage="instruction",
        duration_sec=1.25,
        success=True,
        attempt=1,
        status="ok",
    )
    tracker.record(
        0,
        0,
        model="model",
        task_name="task_00001",
        stage="tests",
        duration_sec=2.5,
        success=False,
        attempt=2,
        status="timeout",
        error="request timeout",
    )

    data = json.loads(usage_path.read_text(encoding="utf-8"))
    assert data["call_count"] == 2
    assert data["success_count"] == 1
    assert data["failure_count"] == 1
    assert data["total_tokens"] == 15
    assert data["by_stage"]["instruction"]["total_tokens"] == 15
    assert data["by_stage"]["instruction"]["success_count"] == 1
    assert data["by_stage"]["tests"]["failure_count"] == 1
    assert data["records"][0]["stage"] == "instruction"
    assert data["records"][1]["status"] == "timeout"


def test_generation_steps_parse_llm_responses(tmp_path: Path) -> None:
    generator = TaskGenerator(make_question(), tmp_path)

    generator.llm_call = lambda *args, **kwargs: "```markdown\n# Task\nDo work.\n```"
    assert generator._generate_instruction() == "# Task\nDo work."

    generator.llm_call = lambda *args, **kwargs: (
        '```json\n{"files": {"task_file/input.txt": "data"}, '
        '"directories": ["task_file"]}\n```'
    )
    env_data = generator._generate_environment("instruction")
    assert env_data["files"]["task_file/input.txt"] == "data"

    generator.llm_call = lambda *args, **kwargs: "```bash\n#!/bin/bash\necho ok\n```"
    assert generator._generate_solution("instruction", env_data) == "#!/bin/bash\necho ok"

    generator.llm_call = lambda *args, **kwargs: (
        "```dockerfile\nFROM ubuntu:22.04\nWORKDIR /app\nCOPY ./task_file /app/task_file\n```"
    )
    dockerfile = generator._generate_dockerfile("instruction")
    assert dockerfile.startswith("FROM python:3.12-slim-bookworm AS verifier_python")
    assert "FROM ubuntu:22.04" in dockerfile
    assert "COPY --from=verifier_python /usr/local /usr/local" in dockerfile

    assert generator._assess_difficulty("instruction") == "easy"


def test_generate_rewrites_instruction_only_after_downstream_stages(tmp_path: Path) -> None:
    generator = TaskGenerator(
        make_question(),
        tmp_path,
        index=3,
        instruction_rewrite_mode="lossy",
    )
    original = "Verbose generated instruction"
    rewritten = "I have a broken data-processing task. Inspect the provided files and fix it."
    captured: dict[str, str] = {}

    def generate_instruction() -> str:
        return original

    def rewrite_instruction(instruction: str) -> str:
        assert instruction == original
        captured["rewrite"] = instruction
        return rewritten

    generator._generate_instruction = generate_instruction
    generator._rewrite_instruction = rewrite_instruction

    def generate_environment(instruction: str) -> dict:
        captured["environment"] = instruction
        return {"files": {}, "directories": ["task_file"]}

    def generate_solution(instruction: str, env_data: dict) -> str:
        del env_data
        captured["solution"] = instruction
        return "#!/usr/bin/env bash\ntrue\n"

    def generate_tests(instruction: str, env_data: dict, solution: str) -> dict:
        del env_data, solution
        captured["tests"] = instruction
        return {"test_outputs_py": "def test_ok():\n    assert True\n"}

    def generate_dockerfile(instruction: str, *args, **kwargs) -> str:
        del args, kwargs
        captured["dockerfile"] = instruction
        return "FROM ubuntu:22.04\nWORKDIR /app\nCOPY ./task_file /app/task_file\n"

    generator._generate_environment = generate_environment
    generator._generate_solution = generate_solution
    generator._generate_tests = generate_tests
    generator._generate_dockerfile = generate_dockerfile

    assert generator.generate() is True
    assert captured == {
        "environment": original,
        "solution": original,
        "tests": original,
        "dockerfile": original,
        "rewrite": original,
    }
    task_dir = tmp_path / "task_00003"
    assert (task_dir / "instruction.md").read_text() == rewritten
    assert 'instruction_rewrite_mode = "lossy"' in (task_dir / "task.toml").read_text()


def test_generate_dockerfile_prompt_includes_generated_artifacts(tmp_path: Path) -> None:
    generator = TaskGenerator(make_question(tags=["node.js", "npm"]), tmp_path)
    captured: dict[str, str] = {}

    def fake_call(prompt: str, *args, **kwargs) -> str:
        captured["prompt"] = prompt
        return (
            "```dockerfile\nFROM node:20-slim\nWORKDIR /app\n"
            "COPY ./package.json /app/package.json\n"
            "COPY ./task_file /app/task_file\n```"
        )

    generator.llm_call = fake_call

    dockerfile = generator._generate_dockerfile(
        "Build the package.",
        {
            "directories": ["task_file/input"],
            "files": {
                "task_file/input/data.json": "{}",
                "package.json": "{\"scripts\":{\"test\":\"node index.js\"}}",
            },
        },
        "#!/bin/bash\nnpm install\nnode index.js\n",
        {"test_outputs_py": "def test_ok():\n    assert True\n"},
    )

    prompt = captured["prompt"]
    assert "[file] /app/package.json" in prompt
    assert "npm install" in prompt
    assert "def test_ok()" in prompt
    assert "FROM node:20-slim" in dockerfile
    assert "COPY --from=verifier_python /usr/local /usr/local" in dockerfile


def test_review_dockerfile_rejects_invalid_patterns() -> None:
    review = review_dockerfile(
        """FROM ubuntu:22.04
WORKDIR /app
RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY ./missing /app/missing
""",
        {"directories": ["task_file"], "files": {"task_file/input.txt": "x"}},
    )

    assert not review["pass"]
    assert "empty apt-get install command" in review["issues"]
    assert "Dockerfile must not install or run uv/uvx test-runtime tooling" in review["issues"]
    assert any(issue.startswith("Dockerfile must include Python 3.12+ and pytest") for issue in review["issues"])
    assert any(issue.startswith("COPY references missing build-context paths") for issue in review["issues"])


def test_review_dockerfile_accepts_runtime_image_with_artifacts() -> None:
    dockerfile = ensure_verifier_deps(
        """FROM node:20-slim
WORKDIR /app
COPY ./package.json /app/package.json
COPY ./task_file /app/task_file
"""
    )

    review = review_dockerfile(
        dockerfile,
        {
            "directories": ["task_file"],
            "files": {
                "task_file/input.json": "{}",
                "package.json": "{\"scripts\":{\"test\":\"node index.js\"}}",
            },
        },
    )

    assert review == {"pass": True, "issues": []}


def test_generate_dockerfile_retries_static_review(tmp_path: Path) -> None:
    generator = TaskGenerator(make_question(tags=["node.js"]), tmp_path)
    prompts = []
    responses = iter(
        [
            "```dockerfile\nFROM ubuntu:22.04\nWORKDIR /app\nCOPY ./missing /app/missing\n```",
            "```dockerfile\nFROM node:20-slim\nWORKDIR /app\nCOPY ./task_file /app/task_file\n```",
        ]
    )

    def fake_call(prompt: str, *args, **kwargs) -> str:
        prompts.append(prompt)
        return next(responses)

    generator.llm_call = fake_call

    dockerfile = generator._generate_dockerfile(
        "Run the Node script.",
        {"directories": ["task_file"], "files": {"task_file/input.txt": "x"}},
        "#!/bin/bash\nnode script.js\n",
        {"test_outputs_py": "def test_ok():\n    assert True\n"},
    )

    assert dockerfile is not None
    assert "FROM node:20-slim" in dockerfile
    assert len(prompts) == 2
    assert "Previous Dockerfile attempt was rejected" in prompts[1]


def test_assess_difficulty_is_deterministic_without_llm(tmp_path: Path) -> None:
    assert TaskGenerator(
        make_question(tags=["json"], categories=["data-processing"], score=600),
        tmp_path,
    )._assess_difficulty("Short JSON conversion task") == "easy"

    assert TaskGenerator(
        make_question(tags=["cuda"], categories=["machine-learning"]),
        tmp_path,
    )._assess_difficulty("Check the installed CUDA version") == "easy"

    assert TaskGenerator(
        make_question(
            body="<p>" + ("x" * 2000) + "</p>",
            accepted_answer={"answer_id": 456, "body": "<p>answer</p>", "score": 99},
            tags=["python"],
            categories=["software-engineering"],
        ),
        tmp_path,
    )._assess_difficulty("Implement a multi-step script") == "medium"

    assert TaskGenerator(
        make_question(
            body="<p>" + ("x" * 5000) + "</p>",
            accepted_answer={"answer_id": 456, "body": "<p>answer</p>", "score": 99},
            tags=["bash"],
            categories=["system-administration"],
        ),
        tmp_path,
    )._assess_difficulty("Inspect a long shell task") == "hard"

    assert TaskGenerator(
        make_question(
            body="<pre><code>a</code></pre>" * 4,
            accepted_answer={"answer_id": 456, "body": "<p>answer</p>", "score": 99},
            tags=["json"],
            categories=["data-processing"],
        ),
        tmp_path,
    )._assess_difficulty("Inspect several snippets") == "hard"


def test_generate_environment_and_dockerfile_fallbacks(tmp_path: Path) -> None:
    generator = TaskGenerator(make_question(), tmp_path)

    generator.llm_call = lambda *args, **kwargs: "not json"
    assert generator._generate_environment("instruction") == {"files": {}, "directories": ["task_file"]}
    assert generator._generate_dockerfile("instruction") is None


def test_ensure_verifier_deps_preserves_non_python_runtime_base_images() -> None:
    for base_image in ("node:20-slim", "golang:1.21-bookworm"):
        dockerfile = f"""FROM {base_image}
WORKDIR /app
COPY ./task_file /app/task_file
"""

        patched = ensure_verifier_deps(dockerfile)

        assert patched.startswith("FROM python:3.12-slim-bookworm AS verifier_python")
        assert f"FROM {base_image}" in patched
        assert "COPY --from=verifier_python /usr/local /usr/local" in patched
        assert patched.index(f"FROM {base_image}") < patched.index("COPY --from=verifier_python")


def test_generate_tests_retries_syntax_and_uses_static_review(tmp_path: Path) -> None:
    generator = TaskGenerator(make_question(), tmp_path)
    responses = iter(
        [
            json.dumps({"test_sh": "curl https://astral.sh/uv/install.sh | sh", "test_outputs_py": "def broken("}),
            json.dumps({"test_sh": "uvx pytest", "test_outputs_py": "def test_ok():\n    assert True\n"}),
        ]
    )
    generator.llm_call = lambda *args, **kwargs: next(responses)

    test_data = generator._generate_tests(
        "instruction",
        {"files": {}, "directories": []},
        "#!/bin/bash\ntrue",
    )

    assert test_data == {"test_sh": STATIC_TEST_SH, "test_outputs_py": "def test_ok():\n    assert True\n"}
    assert "curl" not in test_data["test_sh"]
    assert "uvx" not in test_data["test_sh"]


def test_ensure_verifier_deps_inserts_before_task_files_copy() -> None:
    dockerfile = """FROM ubuntu:22.04
WORKDIR /app
RUN apt-get update && apt-get install -y gcc gdb && rm -rf /var/lib/apt/lists/*
COPY ./task_file /app/task_file
"""

    patched = ensure_verifier_deps(dockerfile)

    assert patched.startswith("FROM python:3.12-slim-bookworm AS verifier_python")
    assert "RUN python -m pip install --no-cache-dir pytest" in patched
    assert "COPY --from=verifier_python /usr/local /usr/local" in patched
    assert patched.index("COPY --from=verifier_python") < patched.index("COPY ./task_file")


def test_ensure_verifier_deps_adds_pytest_to_python312_base() -> None:
    dockerfile = """FROM python:3.12-slim-bookworm
WORKDIR /app
COPY ./task_file /app/task_file
"""

    patched = ensure_verifier_deps(dockerfile)

    assert patched.startswith("FROM python:3.12-slim-bookworm")
    assert "AS verifier_python" not in patched
    assert "RUN python -m pip install --no-cache-dir pytest" in patched
    assert patched.index("pip install") < patched.index("COPY ./task_file")


def test_static_test_runner_is_offline() -> None:
    runner = STATIC_TEST_SH

    assert "curl" not in runner
    assert "uvx" not in runner
    assert "astral.sh" not in runner
    assert "github.com" not in runner
    assert "pip install" not in runner
    assert "apt-get" not in runner
    assert "python3 -m pytest" in runner or "-m pytest" in runner
    assert "python3.13" in runner
    assert "python3.12" in runner
    assert "sys.version_info >= (3, 12)" in runner
    assert "/usr/local/bin/python3" in runner
    assert runner.index("python3.13") < runner.index("/usr/local/bin/python3")
    assert runner.index("/usr/local/bin/python3") < runner.index("/usr/bin/python3")


def test_parse_test_outputs_accepts_python_312_f_string_syntax() -> None:
    code = 'def test_warning_message():\n    warning_lines = ["x"]\n    assert False, f"warnings:\\n{\'\\n\'.join(warning_lines)}"\n'

    tree = parse_test_outputs_py(code)

    assert isinstance(tree, ast.Module)


def test_review_tests_is_static_without_llm(tmp_path: Path) -> None:
    generator = TaskGenerator(make_question(), tmp_path)

    assert generator._review_tests("i", "env", "s", "import os\n\ndef test_ok():\n    assert True\n") == {
        "pass": True,
        "issues": [],
    }

    rerun_review = generator._review_tests(
        "i",
        "env",
        "s",
        'import subprocess\n\ndef test_bad():\n    subprocess.run(["bash", "solve.sh"])\n',
    )
    assert rerun_review["pass"] is False
    assert "test_outputs.py must not execute solve.sh" in rerun_review["issues"]

    import_review = generator._review_tests("i", "env", "s", "import pandas\n")
    assert import_review["pass"] is False
    assert "non-stdlib import in test_outputs.py: pandas" in import_review["issues"]

    py312_syntax_review = generator._review_tests(
        "i",
        "env",
        "s",
        'def test_warning_message():\n    warning_lines = ["x"]\n    assert False, f"warnings:\\n{\'\\n\'.join(warning_lines)}"\n',
    )
    assert py312_syntax_review == {"pass": True, "issues": []}


def test_generate_task_toml_and_write_files(tmp_path: Path) -> None:
    generator = TaskGenerator(make_question(), tmp_path, index=7)
    toml_text = generator._generate_task_toml("hard")

    assert 'difficulty = "hard"' in toml_text
    assert 'category = "system-administration"' in toml_text
    assert "source_question_id = 123" in toml_text
    assert 'source_url = "https://stackoverflow.com/questions/123"' in toml_text
    assert 'instruction_rewrite_mode = "off"' in toml_text

    generator._write_files(
        "# Task",
        {
            "directories": ["task_file/input"],
            "files": {"task_file/input/data.txt": "data"},
        },
        {"test_sh": "#!/bin/bash\ntrue\n", "test_outputs_py": "def test_ok():\n    assert True\n"},
        "#!/bin/bash\necho solved\n",
        "FROM ubuntu:22.04\n",
        "hard",
    )

    task_dir = tmp_path / "task_00007"
    assert (task_dir / "instruction.md").read_text(encoding="utf-8") == "# Task"
    written_dockerfile = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert written_dockerfile.startswith("FROM python:3.12-slim-bookworm AS verifier_python")
    assert "FROM ubuntu:22.04" in written_dockerfile
    assert "COPY --from=verifier_python /usr/local /usr/local" in written_dockerfile
    assert "RUN python -m pip install --no-cache-dir pytest" in written_dockerfile
    assert (task_dir / "environment" / "task_file" / "input" / "data.txt").read_text(encoding="utf-8") == "data"
    assert (task_dir / "tests" / "test.sh").read_text(encoding="utf-8") == STATIC_TEST_SH
    assert (task_dir / "tests" / "test_outputs.py").exists()
    assert (task_dir / "solution" / "solve.sh").exists()
    assert (task_dir / "tests" / "test.sh").stat().st_mode & stat.S_IXUSR
    assert (task_dir / "solution" / "solve.sh").stat().st_mode & stat.S_IXUSR


def write_complete_task(task_dir: Path, task_toml: str) -> None:
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")
    (task_dir / "instruction.md").write_text("# Task\n", encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n", encoding="utf-8")
    (task_dir / "solution" / "solve.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (task_dir / "tests" / "test_outputs.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")


def test_collect_existing_question_ids_reads_new_and_old_task_toml(tmp_path: Path) -> None:
    write_complete_task(
        tmp_path / "task_00001",
        'version = "1.0"\n[metadata]\nsource_question_id = 123\n',
    )
    write_complete_task(
        tmp_path / "task_00007",
        'version = "1.0"\n[metadata]\nsource_url = "https://stackoverflow.com/questions/456/foo"\n',
    )
    incomplete = tmp_path / "task_00008"
    incomplete.mkdir()
    (incomplete / "task.toml").write_text(
        'version = "1.0"\n[metadata]\nsource_question_id = 789\n',
        encoding="utf-8",
    )

    existing = collect_existing_question_ids(tmp_path)

    assert existing[123] == tmp_path / "task_00001"
    assert existing[456] == tmp_path / "task_00007"
    assert 789 not in existing


def test_build_task_args_resume_skips_existing_ids_and_uses_next_free_index(tmp_path: Path) -> None:
    write_complete_task(
        tmp_path / "task_00007",
        'version = "1.0"\n[metadata]\nsource_question_id = 123\n',
    )
    write_complete_task(
        tmp_path / "task_00012",
        'version = "1.0"\n[metadata]\nsource_url = "https://stackoverflow.com/questions/456"\n',
    )
    questions = [
        make_question(question_id=123).__dict__,
        make_question(question_id=999, link="https://stackoverflow.com/questions/999").__dict__,
        make_question(question_id=456, link="https://stackoverflow.com/questions/456").__dict__,
        make_question(question_id=1000, link="https://stackoverflow.com/questions/1000").__dict__,
    ]

    task_args, resume_info = build_task_args(questions, tmp_path, resume=True)

    assert resume_info["skipped_existing_questions"] == 2
    assert resume_info["next_task_index"] == 13
    assert [arg[0]["question_id"] for arg in task_args] == [999, 1000]
    assert [arg[3] for arg in task_args] == [13, 14]


def test_process_question_updates_counters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyGenerator:
        def __init__(self, question, output_dir, index=None):
            self.task_name = f"task_{index:05d}"

        def generate(self) -> bool:
            return True

    before = cli.progress_counter.value
    monkeypatch.setattr(cli, "TaskGenerator", DummyGenerator)

    assert cli.process_question((make_question().__dict__, tmp_path, 1, 1)) is True
    assert cli.progress_counter.value == before + 1


def test_run_generation_notifies_before_all_tasks_finish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from threading import Event

    seed = tmp_path / "seed.json"
    seed.write_text(
        json.dumps(
            {
                "questions": [
                    {"question_id": 1, "accepted_answer": {"body": "a"}},
                    {"question_id": 2, "accepted_answer": {"body": "b"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    first_callback = Event()
    callback_paths: list[Path] = []

    def fake_process_question(task_args: tuple) -> bool:
        if task_args[3] == 1:
            assert first_callback.wait(timeout=2)
        return True

    def on_generated(task_dir: Path, total: int) -> None:
        callback_paths.append(task_dir)
        assert total == 2
        first_callback.set()

    monkeypatch.setattr(cli, "process_question", fake_process_question)
    summary = cli.run_generation(
        input_path=seed,
        output_dir=tmp_path / "candidates",
        workers=2,
        on_generated=on_generated,
    )

    assert summary["success_count"] == 2
    assert {path.name for path in callback_paths} == {"task_00000", "task_00001"}


def test_run_generation_preserves_per_task_exception_isolation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text(
        json.dumps(
            {
                "questions": [
                    {"question_id": 1, "accepted_answer": {"body": "a"}},
                    {"question_id": 2, "accepted_answer": {"body": "b"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    callback_paths: list[Path] = []

    def fake_process_question(task_args: tuple) -> bool:
        if task_args[3] == 0:
            raise RuntimeError("one broken task")
        return True

    monkeypatch.setattr(cli, "process_question", fake_process_question)
    summary = cli.run_generation(
        input_path=seed,
        output_dir=tmp_path / "candidates",
        workers=2,
        on_generated=lambda task_dir, _total: callback_paths.append(task_dir),
    )

    assert summary["success_count"] == 1
    assert summary["error_count"] == 0
    assert [path.name for path in callback_paths] == ["task_00001"]
