from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from generator import task_generator as tg


def make_question(**overrides) -> tg.SOQuestion:
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
    return tg.SOQuestion.from_dict(data)


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

    cleaned = tg.clean_html(html)

    assert "**world**" in cleaned
    assert '```\nprint("<x>")\n```' in cleaned
    assert "[docs](https://example.com)" in cleaned
    assert "`jq`" in cleaned
    assert "- one" in cleaned
    assert "- *two*" in cleaned


def test_truncation_detection_for_json_and_fenced_blocks() -> None:
    assert tg._is_truncated("{" + '"x": ' + '"y"' * 80)
    assert tg._is_truncated("prefix ```json\n" + '{"x": 1}' * 20)
    assert not tg._is_truncated("short")
    assert not tg._is_truncated('```json\n{"x": 1}\n```')


def test_call_llm_api_omits_optional_generation_params_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def fake_post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return DummyResponse()

    monkeypatch.setattr(tg, "token_tracker", DummyTracker())
    monkeypatch.setattr(tg, "_last_request_time", 0)
    monkeypatch.setitem(tg._config, "api_base", "http://example.test/v1")
    monkeypatch.setitem(tg._config, "api_key", "EMPTY")
    monkeypatch.setitem(tg._config, "model", "test-model")
    monkeypatch.setattr(tg.requests, "post", fake_post)

    assert tg.call_llm_api("prompt") == "ok"
    assert "max_tokens" not in payloads[-1]
    assert "temperature" not in payloads[-1]

    assert tg.call_llm_api("prompt", max_tokens=123, temperature=0.4) == "ok"
    assert payloads[-1]["max_tokens"] == 123
    assert payloads[-1]["temperature"] == 0.4


def test_task_name_generation_and_env_file_format(tmp_path: Path) -> None:
    generator = tg.TaskGenerator(make_question(title="A/B: Parse JSON?"), tmp_path)

    assert generator.task_name == "ab-parse-json-123"
    assert generator.guid == tg.hashlib.md5(b"123").hexdigest()[:8]

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
        "system.md": tg.SYSTEM_PROMPT,
        "instruction.md": tg.INSTRUCTION_PROMPT_TEMPLATE,
        "environment.md": tg.ENVIRONMENT_PROMPT_TEMPLATE,
        "solution.md": tg.SOLUTION_PROMPT_TEMPLATE,
        "tests.md": tg.TEST_PROMPT_TEMPLATE,
        "dockerfile.md": tg.DOCKERFILE_PROMPT_TEMPLATE,
    }

    assert tg.PROMPT_TEMPLATE_DIR.name == "task_generator"
    for filename, prompt in prompt_files.items():
        assert prompt == (tg.PROMPT_TEMPLATE_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


def test_token_tracker_records_stage_telemetry(tmp_path: Path) -> None:
    usage_path = tmp_path / "usage.json"
    tracker = tg.TokenTracker(str(usage_path))

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


def test_generation_steps_parse_llm_responses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = tg.TaskGenerator(make_question(), tmp_path)

    monkeypatch.setattr(
        tg,
        "call_llm_api",
        lambda *args, **kwargs: "```markdown\n# Task\nDo work.\n```",
    )
    assert generator._generate_instruction() == "# Task\nDo work."

    monkeypatch.setattr(
        tg,
        "call_llm_api",
        lambda *args, **kwargs: (
            '```json\n{"files": {"task_file/input.txt": "data"}, '
            '"directories": ["task_file"]}\n```'
        ),
    )
    env_data = generator._generate_environment("instruction")
    assert env_data["files"]["task_file/input.txt"] == "data"

    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: "```bash\n#!/bin/bash\necho ok\n```")
    assert generator._generate_solution("instruction", env_data) == "#!/bin/bash\necho ok"

    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: "```dockerfile\nFROM ubuntu:22.04\n```")
    dockerfile = generator._generate_dockerfile("instruction")
    assert dockerfile.startswith("FROM python:3.12-slim-bookworm AS verifier_python")
    assert "FROM ubuntu:22.04" in dockerfile
    assert "COPY --from=verifier_python /usr/local /usr/local" in dockerfile

    assert generator._assess_difficulty("instruction") == "easy"


def test_assess_difficulty_is_deterministic_without_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_call(*args, **kwargs):
        raise AssertionError("difficulty should not call the LLM")

    monkeypatch.setattr(tg, "call_llm_api", fail_call)

    assert tg.TaskGenerator(
        make_question(tags=["json"], categories=["data-processing"], score=600),
        tmp_path,
    )._assess_difficulty("Short JSON conversion task") == "easy"

    assert tg.TaskGenerator(
        make_question(tags=["cuda"], categories=["machine-learning"]),
        tmp_path,
    )._assess_difficulty("Check the installed CUDA version") == "easy"

    assert tg.TaskGenerator(
        make_question(
            body="<p>" + ("x" * 2000) + "</p>",
            accepted_answer={"answer_id": 456, "body": "<p>answer</p>", "score": 99},
            tags=["python"],
            categories=["software-engineering"],
        ),
        tmp_path,
    )._assess_difficulty("Implement a multi-step script") == "medium"

    assert tg.TaskGenerator(
        make_question(
            body="<p>" + ("x" * 5000) + "</p>",
            accepted_answer={"answer_id": 456, "body": "<p>answer</p>", "score": 99},
            tags=["bash"],
            categories=["system-administration"],
        ),
        tmp_path,
    )._assess_difficulty("Inspect a long shell task") == "hard"

    assert tg.TaskGenerator(
        make_question(
            body="<pre><code>a</code></pre>" * 4,
            accepted_answer={"answer_id": 456, "body": "<p>answer</p>", "score": 99},
            tags=["json"],
            categories=["data-processing"],
        ),
        tmp_path,
    )._assess_difficulty("Inspect several snippets") == "hard"


def test_generate_environment_and_dockerfile_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = tg.TaskGenerator(make_question(), tmp_path)

    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: "not json")
    assert generator._generate_environment("instruction") == {"files": {}, "directories": ["task_file"]}
    dockerfile = generator._generate_dockerfile("instruction")
    assert dockerfile.startswith("FROM python:3.12-slim-bookworm")
    assert "python -m pip install --no-cache-dir pytest" in dockerfile


def test_generate_tests_retries_syntax_and_uses_static_review(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = tg.TaskGenerator(make_question(), tmp_path)
    responses = iter(
        [
            json.dumps({"test_sh": "curl https://astral.sh/uv/install.sh | sh", "test_outputs_py": "def broken("}),
            json.dumps({"test_sh": "uvx pytest", "test_outputs_py": "def test_ok():\n    assert True\n"}),
        ]
    )
    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: next(responses))

    test_data = generator._generate_tests(
        "instruction",
        {"files": {}, "directories": []},
        "#!/bin/bash\ntrue",
    )

    assert test_data == {"test_sh": tg.STATIC_TEST_SH, "test_outputs_py": "def test_ok():\n    assert True\n"}
    assert "curl" not in test_data["test_sh"]
    assert "uvx" not in test_data["test_sh"]


def test_ensure_verifier_deps_inserts_before_task_files_copy() -> None:
    dockerfile = """FROM ubuntu:22.04
WORKDIR /app
RUN apt-get update && apt-get install -y gcc gdb && rm -rf /var/lib/apt/lists/*
COPY ./task_file /app/task_file
"""

    patched = tg._ensure_verifier_deps(dockerfile)

    assert patched.startswith("FROM python:3.12-slim-bookworm AS verifier_python")
    assert "RUN python -m pip install --no-cache-dir pytest" in patched
    assert "COPY --from=verifier_python /usr/local /usr/local" in patched
    assert patched.index("COPY --from=verifier_python") < patched.index("COPY ./task_file")


def test_ensure_verifier_deps_adds_pytest_to_python312_base() -> None:
    dockerfile = """FROM python:3.12-slim-bookworm
WORKDIR /app
COPY ./task_file /app/task_file
"""

    patched = tg._ensure_verifier_deps(dockerfile)

    assert patched.startswith("FROM python:3.12-slim-bookworm")
    assert "AS verifier_python" not in patched
    assert "RUN python -m pip install --no-cache-dir pytest" in patched
    assert patched.index("pip install") < patched.index("COPY ./task_file")


def test_static_test_runner_is_offline() -> None:
    runner = tg.STATIC_TEST_SH

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

    tree = tg._parse_test_outputs_py(code)

    assert isinstance(tree, tg.ast.Module)


def test_review_tests_is_static_without_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_call(*args, **kwargs):
        raise AssertionError("test review should not call the LLM")

    generator = tg.TaskGenerator(make_question(), tmp_path)
    monkeypatch.setattr(tg, "call_llm_api", fail_call)

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
    generator = tg.TaskGenerator(make_question(), tmp_path, index=7)
    toml_text = generator._generate_task_toml("hard")

    assert 'difficulty = "hard"' in toml_text
    assert 'category = "system-administration"' in toml_text
    assert "source_question_id = 123" in toml_text
    assert 'source_url = "https://stackoverflow.com/questions/123"' in toml_text

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
    assert (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8") == "FROM ubuntu:22.04\n"
    assert (task_dir / "environment" / "task_file" / "input" / "data.txt").read_text(encoding="utf-8") == "data"
    assert (task_dir / "tests" / "test.sh").read_text(encoding="utf-8") == tg.STATIC_TEST_SH
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

    existing = tg._collect_existing_question_ids(tmp_path)

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

    task_args, resume_info = tg._build_task_args(questions, tmp_path, resume=True)

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

    before = tg.progress_counter.value
    monkeypatch.setattr(tg, "TaskGenerator", DummyGenerator)

    assert tg.process_question((make_question().__dict__, tmp_path, 1, 1)) is True
    assert tg.progress_counter.value == before + 1
