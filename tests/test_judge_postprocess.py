from __future__ import annotations

from app.graph.judge_models import CoverageItem, JudgeLLMOutput, PublicReport
from app.graph.nodes.judge import COVERAGE_KEYS, build_judge_response


def _base_llm_output(*, acc: int, comp: int, flags: list[str] | None = None) -> JudgeLLMOutput:
    return JudgeLLMOutput(
        factual_accuracy={
            "score": acc,
            "reasoning": "Reasoning text.",
            "strengths": ["Supported by sources"],
            "weaknesses": [],
        },
        completeness={
            "score": comp,
            "reasoning": "Completeness reasoning.",
            "strengths": [],
            "weaknesses": [],
            "coverage": {k: CoverageItem(status="covered", notes="") for k in COVERAGE_KEYS},
        },
        overall_assessment="A short overall assessment paragraph with enough detail.",
        flags=flags or [],
        suggested_improvements=["Add citations for key claims."],
        confidence=0.7,
    )


def test_schema_builds_and_percentages() -> None:
    llm_out = _base_llm_output(acc=8, comp=7)
    report = PublicReport(
        id="rep_abc123",
        topic="Test",
        content="Some content",
        sources=[],
        word_count=2,
        created_at=None,
    )

    resp = build_judge_response(report=report, llm_out=llm_out)

    assert resp.status == "complete"
    assert resp.evaluation.factual_accuracy.score == 8
    assert resp.evaluation.factual_accuracy.percentage == 80
    assert resp.evaluation.completeness.score == 7
    assert resp.evaluation.completeness.percentage == 70

    # overall_score must be strict arithmetic mean (rounded to 1 decimal)
    assert resp.evaluation.overall_score == 7.5

    # coverage must include all template keys
    assert set(resp.evaluation.completeness.coverage.keys()) == set(COVERAGE_KEYS)

    assert isinstance(resp.metadata.evaluation_id, str)
    assert isinstance(resp.metadata.evaluated_at, str)
    assert isinstance(resp.metadata.processing_time_ms, int) or resp.metadata.processing_time_ms is None


def test_gating_accurate_but_incomplete_revise() -> None:
    llm_out = _base_llm_output(acc=8, comp=5)
    report = PublicReport(
        id="rep_abc123",
        topic="Test",
        content="Some content",
        sources=[],
        word_count=2,
        created_at=None,
    )
    resp = build_judge_response(report=report, llm_out=llm_out)

    assert resp.evaluation.factual_accuracy.score >= 7
    assert resp.evaluation.completeness.score < 7
    assert resp.evaluation.recommendation == "revise"


def test_gating_complete_but_wrong_not_publish() -> None:
    llm_out = _base_llm_output(acc=4, comp=8, flags=["contradiction-detected"])
    report = PublicReport(
        id="rep_abc123",
        topic="Test",
        content="Some content",
        sources=[],
        word_count=2,
        created_at=None,
    )
    resp = build_judge_response(report=report, llm_out=llm_out)

    assert resp.evaluation.factual_accuracy.score < 7
    assert resp.evaluation.recommendation in {"revise", "reject"}
    assert resp.evaluation.recommendation != "publish"


def test_scores_are_clamped_and_int() -> None:
    # Simulate a model or upstream bug producing out-of-range numbers.
    # We validate that build_judge_response clamps defensively.
    raw = {
        "factual_accuracy": {
            "score": 99,
            "reasoning": "Reasoning text.",
            "strengths": ["Supported by sources"],
            "weaknesses": [],
        },
        "completeness": {
            "score": -2,
            "reasoning": "Completeness reasoning.",
            "strengths": [],
            "weaknesses": [],
            "coverage": {k: {"status": "covered", "notes": ""} for k in COVERAGE_KEYS},
        },
        "overall_assessment": "A short overall assessment paragraph with enough detail.",
        "flags": [],
        "suggested_improvements": ["Add citations for key claims."],
        "confidence": 0.7,
    }

    llm_out = JudgeLLMOutput.model_validate(raw)

    report = PublicReport(
        id="rep_abc123",
        topic="Test",
        content="Some content",
        sources=[],
        word_count=2,
        created_at=None,
    )
    resp = build_judge_response(report=report, llm_out=llm_out)

    assert resp.evaluation.factual_accuracy.score == 10
    assert resp.evaluation.completeness.score == 0


def test_overall_assessment_non_empty() -> None:
    llm_out = _base_llm_output(acc=7, comp=7)
    report = PublicReport(
        id="rep_abc123",
        topic="Test",
        content="Some content",
        sources=[],
        word_count=2,
        created_at=None,
    )
    resp = build_judge_response(report=report, llm_out=llm_out)

    assert isinstance(resp.evaluation.overall_assessment, str)
    assert resp.evaluation.overall_assessment.strip()


def test_overall_score_rounding_is_consistent_one_decimal() -> None:
    # 9 and 8 average to 8.5 exactly; enforce 1-decimal float (not 2-decimal legacy)
    llm_out = _base_llm_output(acc=9, comp=8)
    report = PublicReport(id="rep_abc123", topic="Test", content="Some content", sources=[], word_count=2, created_at=None)
    resp = build_judge_response(report=report, llm_out=llm_out)

    assert resp.evaluation.overall_score == 8.5
    assert resp.evaluation.grade == "B"
