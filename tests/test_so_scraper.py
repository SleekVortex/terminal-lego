from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scraper import so_scraper as scraper


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.stderr = ""
        self.stdout = ""

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_clean_html_removes_tags_and_unescapes_entities() -> None:
    assert scraper.clean_html("<p>A &lt;b&gt;&amp;&nbsp;&#39;x&#39;</p>") == "A <b>& 'x'"


def test_save_data_and_load_existing_ids(tmp_path: Path) -> None:
    questions = [
        {
            "question_id": 1,
            "title": "Title",
            "link": "https://example.com/q/1",
            "score": 3,
            "tags": ["bash"],
            "search_tag": "bash",
            "body": "<p>body</p>",
            "accepted_answer": {"body": "<p>answer</p>"},
        }
    ]
    output_json = tmp_path / "so_data.json"
    output_jsonl = tmp_path / "so_data.jsonl"

    scraper.save_data(questions, str(output_json), str(output_jsonl))

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["metadata"]["total"] == 1
    assert saved["metadata"]["with_answers"] == 1
    row = json.loads(output_jsonl.read_text(encoding="utf-8"))
    assert row["body_text"] == "body"
    assert row["answer_text"] == "answer"
    assert scraper.load_existing_ids(tmp_path) == {1}


def test_get_questions_by_tag_filters_for_accepted_answers_and_dedup(monkeypatch) -> None:
    payload = {
        "items": [
            {
                "question_id": 1,
                "title": "keep",
                "link": "https://example.com/q/1",
                "score": 10,
                "view_count": 100,
                "answer_count": 2,
                "accepted_answer_id": 11,
                "tags": ["bash"],
                "body": "<p>body</p>",
                "creation_date": 1,
            },
            {
                "question_id": 2,
                "title": "no accepted answer",
                "link": "https://example.com/q/2",
                "score": 10,
                "tags": ["bash"],
            },
            {
                "question_id": 3,
                "title": "duplicate",
                "link": "https://example.com/q/3",
                "score": 10,
                "accepted_answer_id": 33,
                "tags": ["bash"],
            },
        ],
        "quota_remaining": 500,
        "has_more": False,
    }

    def fake_get(url, params, timeout):
        assert url.endswith("/questions")
        assert params["tagged"] == "bash"
        return FakeResponse(payload)

    monkeypatch.setattr(scraper.requests, "get", fake_get)
    monkeypatch.setattr(scraper.time, "sleep", lambda seconds: None)

    questions, should_stop = scraper.get_questions_by_tag(
        "bash",
        api_key="KEY",
        exclude_ids={3},
    )

    assert should_stop is False
    assert [question["question_id"] for question in questions] == [1]
    assert questions[0]["accepted_answer_id"] == 11


def test_get_questions_by_tag_stops_on_low_quota(monkeypatch) -> None:
    def fake_get(url, params, timeout):
        return FakeResponse({"items": [], "quota_remaining": 1, "has_more": True})

    monkeypatch.setattr(scraper.requests, "get", fake_get)
    monkeypatch.setattr(scraper.time, "sleep", lambda seconds: None)

    questions, should_stop = scraper.get_questions_by_tag("bash", api_key="KEY")

    assert questions == []
    assert should_stop is True


def test_get_answers_batch_maps_answer_ids(monkeypatch) -> None:
    def fake_get(url, params, timeout):
        assert url.endswith("/answers/10;20")
        return FakeResponse(
            {
                "backoff": 1,
                "items": [
                    {"answer_id": 10, "body": "<p>a</p>", "score": 2},
                    {"answer_id": 20, "body": "<p>b</p>", "score": 3},
                ],
            }
        )

    monkeypatch.setattr(scraper.requests, "get", fake_get)
    monkeypatch.setattr(scraper.time, "sleep", lambda seconds: None)

    assert scraper.get_answers_batch([10, 20], api_key="KEY") == {
        10: {"answer_id": 10, "body": "<p>a</p>", "score": 2},
        20: {"answer_id": 20, "body": "<p>b</p>", "score": 3},
    }
