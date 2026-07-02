from __future__ import annotations

import json
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
    assert generator._generate_dockerfile("instruction") == "FROM ubuntu:22.04"

    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: "This is medium.")
    assert generator._assess_difficulty("instruction") == "medium"


def test_generate_environment_and_dockerfile_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = tg.TaskGenerator(make_question(), tmp_path)

    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: "not json")
    assert generator._generate_environment("instruction") == {"files": {}, "directories": ["task_file"]}
    assert generator._generate_dockerfile("instruction").startswith("FROM ubuntu:22.04")


def test_generate_tests_retries_syntax_and_uses_review(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = tg.TaskGenerator(make_question(), tmp_path)
    responses = iter(
        [
            json.dumps({"test_sh": "run", "test_outputs_py": "def broken("}),
            json.dumps({"test_sh": "run", "test_outputs_py": "def test_ok():\n    assert True\n"}),
            json.dumps({"pass": True, "issues": []}),
        ]
    )
    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: next(responses))

    test_data = generator._generate_tests(
        "instruction",
        {"files": {}, "directories": []},
        "#!/bin/bash\ntrue",
    )

    assert test_data == {"test_sh": "run", "test_outputs_py": "def test_ok():\n    assert True\n"}


def test_review_tests_parsing_and_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = tg.TaskGenerator(make_question(), tmp_path)

    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: '```json\n{"pass": false, "issues": ["x"]}\n```')
    assert generator._review_tests("i", "env", "s", "code") == {"pass": False, "issues": ["x"]}

    monkeypatch.setattr(tg, "call_llm_api", lambda *args, **kwargs: "The result is \"pass\": true")
    assert generator._review_tests("i", "env", "s", "code") == {"pass": True, "issues": []}


def test_generate_task_toml_and_write_files(tmp_path: Path) -> None:
    generator = tg.TaskGenerator(make_question(), tmp_path, index=7)
    toml_text = generator._generate_task_toml("hard")

    assert 'difficulty = "hard"' in toml_text
    assert 'category = "system-administration"' in toml_text
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
    assert (task_dir / "tests" / "test_outputs.py").exists()
    assert (task_dir / "solution" / "solve.sh").exists()


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
