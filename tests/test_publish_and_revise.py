from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def _mk_eval(*, recommendation: str, factual: int, complete: int) -> dict:
    return {
        "factual_accuracy": {
            "score": factual,
            "max_score": 10,
            "percentage": factual * 10,
            "reasoning": "Sufficient reasoning for test payload.",
            "strengths": [],
            "weaknesses": [],
        },
        "completeness": {
            "score": complete,
            "max_score": 10,
            "percentage": complete * 10,
            "reasoning": "Sufficient reasoning for test payload.",
            "strengths": [],
            "weaknesses": [],
            "coverage": {},
        },
        "overall_score": float((factual + complete) / 2),
        "grade": "A",
        "overall_assessment": "This is a sufficiently long assessment for tests.",
        "recommendation": recommendation,
        "confidence": 0.5,
        "flags": [],
        "suggested_improvements": [],
    }


def test_publish_gating_enforced(tmp_path: Path, monkeypatch):
    # Ensure we write into a temp ./reports directory.
    monkeypatch.chdir(tmp_path)

    from app.main import app

    client = TestClient(app)

    report_id = "rep_test"
    url = f"/api/reports/{report_id}/publish"

    payload = {
        "query": "test",
        "report": {"id": report_id, "topic": "test", "content": "hello", "sources": [], "word_count": 1, "created_at": "2026-01-01T00:00:00Z"},
        "evaluation": _mk_eval(recommendation="revise", factual=10, complete=10),
        "metadata": {"evaluation_id": "e1"},
        "sources": [],
    }

    res = client.post(url, json=payload)
    assert res.status_code == 403
    body = res.json()
    assert "detail" in body


def test_publish_writes_files(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from app.main import app

    client = TestClient(app)

    report_id = "rep_write"
    url = f"/api/reports/{report_id}/publish"

    payload = {
        "query": "test",
        "report": {"id": report_id, "topic": "test", "content": "hello", "sources": [], "word_count": 1, "created_at": "2026-01-01T00:00:00Z"},
        "evaluation": _mk_eval(recommendation="publish", factual=7, complete=7),
        "metadata": {"evaluation_id": "e1"},
        "sources": [{"title": "Example", "url": "https://example.com"}],
    }

    res = client.post(url, json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["ok"] is True
    assert data["report_id"] == report_id
    assert "published_at" in data
    assert "artifacts" in data

    md_path = Path(data["artifacts"]["md"])
    json_path = Path(data["artifacts"]["json"])

    assert md_path.exists(), f"missing md artifact: {md_path}"
    assert json_path.exists(), f"missing json artifact: {json_path}"

    md_text = md_path.read_text(encoding="utf-8")
    assert "## Sources" in md_text
    assert "example.com" in md_text

    payload_saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload_saved["published_at"] == data["published_at"]
    assert payload_saved["report"]["id"] == report_id


def test_revise_returns_versioned_report_id(tmp_path: Path, monkeypatch):
    # Note: this hits the real graph; if you want faster tests, we can monkeypatch build_research_graph.
    monkeypatch.chdir(tmp_path)

    from app.main import app

    client = TestClient(app)

    report_id = "rep_abc123"
    url = f"/api/reports/{report_id}/revise"

    payload = {
        "query": "test",
        "user_note": "please improve",
        "report": {"id": report_id, "topic": "test", "content": "hello", "sources": [], "word_count": 1, "created_at": "2026-01-01T00:00:00Z"},
        "evaluation": _mk_eval(recommendation="revise", factual=5, complete=5),
        "metadata": {"evaluation_id": "e1"},
        "sources": [],
    }

    res = client.post(url, json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data.get("status") == "complete"
    assert data.get("report")
    assert data["report"]["id"].startswith(report_id)
    assert data["report"]["id"] != report_id
