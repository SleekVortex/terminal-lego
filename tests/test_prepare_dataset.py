from __future__ import annotations

import json
from pathlib import Path

import pytest

from stackoverflow import prepare_dataset as prep


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_iter_jsonl_skips_blank_lines_and_rejects_invalid_json(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert list(prep.iter_jsonl(good)) == [{"a": 1}, {"b": 2}]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a": 1}\n[]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        list(prep.iter_jsonl(bad))


def test_normalize_question_accepts_xml_style_rows_and_classifies_tags() -> None:
    row = {
        "Id": "123",
        "Title": "How do I parse JSON in bash?",
        "Body": "<p>body</p>",
        "Tags": "<bash><json>",
        "Score": "5",
        "ViewCount": "100",
        "AnswerCount": "2",
        "AcceptedAnswerId": "456",
        "accepted_answer": {"answer_id": 456, "body": "<p>answer</p>", "score": "9"},
    }

    normalized = prep.normalize_question(row)

    assert normalized["question_id"] == 123
    assert normalized["title"] == row["Title"]
    assert normalized["tags"] == ["bash", "json"]
    assert normalized["accepted_answer_id"] == 456
    assert normalized["accepted_answer"]["score"] == "9"
    assert "system-administration" in normalized["categories"]
    assert "data-processing" in normalized["categories"]
    assert normalized["selected_category"] == normalized["categories"][0]
    assert normalized["link"] == "https://stackoverflow.com/questions/123"


def test_normalize_question_rejects_missing_question_id() -> None:
    with pytest.raises(ValueError, match="positive question_id"):
        prep.normalize_question({"title": "missing"})


def test_iter_normalized_filtered_applies_answer_score_and_category_filters(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(
        path,
        [
            {
                "question_id": 1,
                "title": "keep",
                "body": "body",
                "tags": ["bash"],
                "score": 3,
                "accepted_answer": {"body": "answer"},
            },
            {
                "question_id": 2,
                "title": "low score",
                "body": "body",
                "tags": ["bash"],
                "score": -1,
                "accepted_answer": {"body": "answer"},
            },
            {
                "question_id": 3,
                "title": "no answer",
                "body": "body",
                "tags": ["json"],
                "score": 5,
            },
            {
                "question_id": 4,
                "title": "wrong category",
                "body": "body",
                "tags": ["openssl"],
                "score": 5,
                "accepted_answer": {"body": "answer"},
            },
        ],
    )

    rows = list(
        prep.iter_normalized_filtered(
            path,
            min_score=0,
            categories=["system-administration"],
            allow_uncategorized=False,
        )
    )

    assert [row["question_id"] for row in rows] == [1]


def test_selection_helpers_keep_expected_rows_and_order() -> None:
    rows = [
        {"question_id": 1, "score": 1, "view_count": 10, "accepted_answer": {"score": 1}},
        {"question_id": 2, "score": 5, "view_count": 1, "accepted_answer": {"score": 1}},
        {"question_id": 3, "score": 5, "view_count": 10, "accepted_answer": {"score": 9}},
        {"question_id": 4, "score": 2, "view_count": 100, "accepted_answer": {"score": 0}},
    ]

    assert [row["question_id"] for row in prep.select_first(rows, start=1, limit=2)] == [2, 3]
    assert [row["question_id"] for row in prep.select_top(rows, limit=2)] == [3, 2]


def test_select_per_category_sets_selected_category_and_takes_top_scores() -> None:
    rows = [
        {"question_id": 1, "score": 1, "categories": ["security"], "accepted_answer": {"score": 1}},
        {"question_id": 2, "score": 5, "categories": ["security"], "accepted_answer": {"score": 1}},
        {
            "question_id": 3,
            "score": 9,
            "categories": ["data-processing"],
            "accepted_answer": {"score": 1},
        },
    ]

    selected = prep.select_per_category(rows, per_category=1, categories=["security", "data-processing"])

    assert [(row["question_id"], row["selected_category"]) for row in selected] == [
        (2, "security"),
        (3, "data-processing"),
    ]


def test_select_benchmark_distribution_returns_requested_size_without_duplicates() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "question_id": index,
                "score": index,
                "categories": ["software-engineering" if index % 2 else "security"],
                "accepted_answer": {"score": index},
            }
        )

    selected = prep.select_benchmark_distribution(
        rows,
        sample_size=7,
        categories=["software-engineering", "security"],
    )

    assert len(selected) == 7
    assert len({row["question_id"] for row in selected}) == 7
    assert {row["selected_category"] for row in selected} <= {"software-engineering", "security"}


def test_write_outputs_and_main_jsonl_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "question_id": 1,
                "title": "one",
                "body": "body",
                "tags": ["bash"],
                "score": 1,
                "accepted_answer": {"body": "answer", "score": 2},
            },
            {
                "question_id": 2,
                "title": "two",
                "body": "body",
                "tags": ["json"],
                "score": 10,
                "accepted_answer": {"body": "answer", "score": 1},
            },
        ],
    )

    assert prep.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--format",
            "jsonl",
            "--sort-by-score",
            "--limit",
            "1",
        ]
    ) == 0

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in rows] == [2]
    printed = json.loads(capsys.readouterr().out)
    assert printed["rows"] == 1
    assert printed["output"] == str(output_path)
