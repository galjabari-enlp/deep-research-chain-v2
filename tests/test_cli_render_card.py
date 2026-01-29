from __future__ import annotations

from app.cli_render import render_evaluation_card, render_missing_evaluation_card
from app.graph.judge_models import JudgeEvaluation, PublicReport


def _report() -> PublicReport:
    return PublicReport(
        id="rep_test",
        topic="Mediterranean Diet",
        content="Some content",
        sources=[],
        word_count=2,
        created_at="2026-01-01T00:00:00Z",
    )


def test_render_card_publish_snapshot_contains_key_lines() -> None:
    ev = JudgeEvaluation(
        factual_accuracy={
            "score": 9,
            "max_score": 10,
            "percentage": 90,
            "reasoning": "...",
            "strengths": ["All claims sourced", "Credible references"],
            "weaknesses": [],
        },
        completeness={
            "score": 8,
            "max_score": 10,
            "percentage": 80,
            "reasoning": "...",
            "strengths": ["Covers main benefits"],
            "weaknesses": ["practical meal plans"],
            "coverage": {"practical_guidance": {"status": "not_applicable", "notes": "No practical plans in sources"}},
        },
        overall_score=8.5,
        grade="B",
        overall_assessment="High-quality research with strong evidence base. Minor gap in practical guidance.",
        recommendation="publish",
        confidence=0.72,
        flags=[],
        suggested_improvements=[],
    )

    card = render_evaluation_card(report=_report(), evaluation=ev, width=60)

    assert "Overall Quality:" in card
    assert "8.5/10" in card
    assert "Factual Accuracy" in card
    assert "Completeness" in card
    assert "Judge's Assessment" in card
    assert "APPROVED FOR PUBLICATION" in card
    assert "[Publish Now] [Request Revision]" in card


def test_render_card_revise_status() -> None:
    ev = JudgeEvaluation(
        factual_accuracy={
            "score": 6,
            "max_score": 10,
            "percentage": 60,
            "reasoning": "...",
            "strengths": [],
            "weaknesses": ["Missing citations"],
        },
        completeness={
            "score": 8,
            "max_score": 10,
            "percentage": 80,
            "reasoning": "...",
            "strengths": [],
            "weaknesses": [],
            "coverage": {},
        },
        overall_score=6.7,
        grade="D",
        overall_assessment="Some claims appear unsupported; revise before publishing.",
        recommendation="revise",
        confidence=0.5,
        flags=[],
        suggested_improvements=["Add citations"],
    )

    card = render_evaluation_card(report=_report(), evaluation=ev, width=60)
    assert "NEEDS REVISION" in card


def test_render_missing_evaluation_fallback() -> None:
    card = render_missing_evaluation_card(report=_report(), width=60)
    assert "Judge evaluation unavailable" in card
    assert "[Publish Now] [Request Revision]" in card
