from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List

from app.graph.enforce import model_validate_json_with_retry, sanitize_for_prompt
from app.graph.judge_models import (
    CoverageItem,
    JudgeEvaluation,
    JudgeLLMOutput,
    JudgeMetadata,
    JudgeResponse,
    PublicReport,
    utc_now_iso,
)
from app.graph.models import SearchResult
from app.graph.state import ResearchState
from app.services.llm import LLMClient

logger = logging.getLogger(__name__)


# Stable template checklist for completeness.coverage.
# Frontend can rely on these keys to render a consistent section-by-section view.
COVERAGE_KEYS: list[str] = [
    "overview",
    "key_points",
    "evidence_and_citations",
    "benefits",
    "risks_caveats",
    "limitations",
    "practical_guidance",
    "populations_considerations",
    "conclusion",
]


SYSTEM_PROMPT = (
    "You are Judge, a strict research report evaluator. "
    "Evaluate the final report for factual accuracy and completeness. "
    "Be critical: do not give high scores unless the report is well-supported by the provided sources. "
    "Do not invent sources or facts. Only evaluate based on the report content and the sources list. "
    "Return ONLY valid JSON matching the provided schema. No markdown. No commentary."
)


def _round_percentage(score: int, max_score: int) -> int:
    # Deterministic rounding: nearest int.
    return int(round((score / max_score) * 100))


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _compute_overall_score(*, accuracy: int, completeness: int) -> float:
    """Deterministic overall score formula.

    We weight factual accuracy higher because publishing incorrect information is riskier
    than missing details.

    overall_score = 0.65 * factual_accuracy + 0.35 * completeness
    """

    return round((0.65 * accuracy) + (0.35 * completeness), 2)


def _grade_from_overall(overall: float) -> str:
    if overall >= 9.0:
        return "A"
    if overall >= 8.0:
        return "B"
    if overall >= 7.0:
        return "C"
    if overall >= 6.0:
        return "D"
    return "F"


def _recommendation(
    *,
    accuracy: int,
    completeness: int,
    flags: list[str],
) -> str:
    # Hard gating thresholds
    if accuracy < 7:
        # Escalate to reject if severe safety/credibility flags exist.
        severe = {"unsourced-medical-claims", "policy-risk", "contradiction-detected"}
        if any(f in severe for f in flags) or accuracy <= 4:
            return "reject"
        return "revise"

    if completeness < 7:
        return "revise"

    # Both pass thresholds: publish unless serious flags.
    serious_blockers = {"policy-risk", "unsourced-medical-claims", "contradiction-detected"}
    if any(f in serious_blockers for f in flags):
        return "revise"

    return "publish"


def _normalize_flags(flags: Any) -> list[str]:
    if not isinstance(flags, list):
        return []
    out: list[str] = []
    for f in flags:
        s = str(f or "").strip()
        if not s:
            continue
        out.append(s)
    # de-dupe preserve order
    seen: set[str] = set()
    deduped: list[str] = []
    for f in out:
        if f in seen:
            continue
        seen.add(f)
        deduped.append(f)
    return deduped[:20]


def _default_coverage() -> Dict[str, CoverageItem]:
    return {k: CoverageItem(status="missing", notes="") for k in COVERAGE_KEYS}


def _merge_coverage(raw: Any) -> Dict[str, CoverageItem]:
    base = _default_coverage()
    if not isinstance(raw, dict):
        return base

    for k in COVERAGE_KEYS:
        v = raw.get(k)
        if isinstance(v, dict):
            status = str(v.get("status") or "missing").strip().lower()
            if status not in {"covered", "missing", "not_applicable"}:
                status = "missing"
            notes = str(v.get("notes") or "")
            base[k] = CoverageItem(status=status, notes=notes[:240])
        elif isinstance(v, bool):
            # Back-compat: if an older judge returns boolean, map True->covered, False->missing.
            base[k] = CoverageItem(status=("covered" if v else "missing"), notes="")

    return base


def _public_report_from_state(state: ResearchState) -> PublicReport:
    # If already set by report_node adapter, use it.
    if state.public_report is not None:
        return state.public_report

    # Best-effort mapping from legacy FinalReport.
    report = state.report
    blocks = (report.blocks if report is not None else [])
    content_parts: list[str] = []
    for b in blocks:
        if (b.heading or "").strip():
            content_parts.append(f"{b.heading}\n{b.text}")
        else:
            content_parts.append(b.text)

    content = "\n\n".join([p for p in content_parts if (p or "").strip()])
    if not content and report is not None:
        content = "\n".join(report.key_findings or [])

    # Flatten unique sources from searches.
    sources: list[dict] = []
    seen: set[str] = set()
    for s in state.searches:
        for r in s.results:
            url = str(r.url)
            if url in seen:
                continue
            seen.add(url)
            sources.append({"title": r.title, "url": url, "snippet": r.snippet, "source": r.source})

    word_count = len([w for w in content.split() if w.strip()]) if content else 0

    return PublicReport(
        id=f"rep_{uuid.uuid4().hex[:8]}",
        topic=state.query,
        content=content,
        sources=sources,
        word_count=word_count,
        created_at=None,
    )


def _judge_user_prompt(*, report: PublicReport) -> str:
    schema = JudgeLLMOutput.model_json_schema()

    # Provide the sources list as compact lines for judge evaluation.
    src_lines: list[str] = []
    for i, s in enumerate((report.sources or [])[:25], start=1):
        try:
            title = str(s.get("title") or "")
            url = str(s.get("url") or "")
            snip = str(s.get("snippet") or "").replace("\n", " ")
            if len(snip) > 240:
                snip = snip[:240] + "…"
            src_lines.append(f"[{i}] {title} | {url} | {snip}")
        except Exception:
            continue

    # Ask for stable coverage keys.
    coverage_keys = ", ".join(COVERAGE_KEYS)

    return (
        f"Report topic: {sanitize_for_prompt(report.topic, 600)}\n"
        f"Report content:\n{sanitize_for_prompt(report.content, 12000)}\n\n"
        "Sources (evaluate using these only):\n"
        + "\n".join(src_lines)
        + "\n\n"
        "Task: Evaluate factual accuracy and completeness.\n"
        "For factual accuracy, look for unsourced claims, contradictions, and citation support.\n"
        "For completeness, check whether the report covers the expected aspects.\n\n"
        f"Completeness coverage checklist keys (MUST include all of them): {coverage_keys}\n"
        "For each key, set coverage[key].status to one of: 'covered', 'missing', 'not_applicable'.\n"
        "Use 'not_applicable' only if the section is not relevant for this query OR the provided sources do not reasonably support it.\n\n"
        "Scoring rubric:\n"
        "- factual_accuracy.score: 0-10 (integer). 10 only if claims are strongly supported by the provided sources.\n"
        "- completeness.score: 0-10 (integer). 10 only if all key aspects are covered with adequate depth.\n"
        "Flags: include strings like 'missing-citations', 'unsourced-medical-claims', 'contradiction-detected', 'outdated-sources', 'policy-risk'.\n"
        "Suggested improvements: concrete action items for revision.\n\n"
        "OUTPUT RULES:\n"
        "- Return ONLY JSON.\n"
        "- Must match schema exactly. No extra keys.\n"
        "- completeness.coverage must include all checklist keys.\n\n"
        f"SCHEMA: {schema}"
    )


def build_judge_response(*, report: PublicReport, llm_out: JudgeLLMOutput) -> JudgeResponse:
    # Clamp + compute derived fields
    acc = _clamp_int(llm_out.factual_accuracy.score, 0, 10, 0)
    comp = _clamp_int(llm_out.completeness.score, 0, 10, 0)

    acc_pct = _round_percentage(acc, 10)
    comp_pct = _round_percentage(comp, 10)

    flags = _normalize_flags(llm_out.flags)
    coverage = _merge_coverage(llm_out.completeness.coverage)

    overall = _compute_overall_score(accuracy=acc, completeness=comp)
    grade = _grade_from_overall(overall)

    recommendation = _recommendation(accuracy=acc, completeness=comp, flags=flags)

    confidence = float(llm_out.confidence) if llm_out.confidence is not None else 0.5
    confidence = max(0.0, min(1.0, confidence))

    evaluation = JudgeEvaluation(
        factual_accuracy={
            "score": acc,
            "max_score": 10,
            "percentage": acc_pct,
            "reasoning": llm_out.factual_accuracy.reasoning,
            "strengths": list(llm_out.factual_accuracy.strengths or []),
            "weaknesses": list(llm_out.factual_accuracy.weaknesses or []),
        },
        completeness={
            "score": comp,
            "max_score": 10,
            "percentage": comp_pct,
            "reasoning": llm_out.completeness.reasoning,
            "strengths": list(llm_out.completeness.strengths or []),
            "weaknesses": list(llm_out.completeness.weaknesses or []),
            "coverage": {k: coverage[k] for k in COVERAGE_KEYS},
        },
        overall_score=overall,
        grade=grade,
        overall_assessment=llm_out.overall_assessment,
        recommendation=recommendation,
        confidence=confidence,
        flags=flags,
        suggested_improvements=list(llm_out.suggested_improvements or []),
    )

    metadata = JudgeMetadata(
        evaluation_id=str(uuid.uuid4()),
        evaluated_at=utc_now_iso(),
        judge_model="claude-sonnet-4-20250514",
        processing_time_ms=None,
        evaluation_version="1.0",
    )

    return JudgeResponse(status="complete", report=report, evaluation=evaluation, metadata=metadata)


async def judge_node(state: ResearchState, llm: LLMClient) -> ResearchState:
    start = time.perf_counter()

    if state.report is None and state.public_report is None:
        state.trace.append("Judge skipped: no report")
        return state

    report = _public_report_from_state(state)

    user_prompt = _judge_user_prompt(report=report)
    raw = await llm.chat_json(system=SYSTEM_PROMPT, user=user_prompt)

    parsed, err = model_validate_json_with_retry(JudgeLLMOutput, raw)
    if parsed is None:
        # Re-ask once with error context.
        retry = (
            user_prompt
            + "\n\nYour previous output failed validation. "
            + f"Validation error: {sanitize_for_prompt(err or 'unknown', 1200)}\n"
            + "Return ONLY valid JSON matching the schema."
        )
        raw2 = await llm.chat_json(system=SYSTEM_PROMPT, user=retry)
        parsed, err2 = model_validate_json_with_retry(JudgeLLMOutput, raw2)
        if parsed is None:
            state.trace.append("Judge failed: unable to produce valid JSON")
            logger.warning("Judge failed validation twice. err=%s err2=%s raw=%s", err, err2, raw[:1200])
            return state

    assert isinstance(parsed, JudgeLLMOutput)

    resp = build_judge_response(report=report, llm_out=parsed)

    # Patch processing time into metadata.
    processing_ms = int((time.perf_counter() - start) * 1000)
    resp.metadata.processing_time_ms = processing_ms

    # Persist into state for future revision loop.
    if state.judge_evaluation is not None:
        state.prior_evaluations.append(state.judge_evaluation)

    state.public_report = resp.report
    state.judge_evaluation = resp.evaluation
    state.judge_metadata = resp.metadata

    from app.graph.trace_models import add_trace_item

    add_trace_item(
        state.execution_trace,
        iteration=state.iteration_count,
        section="report",
        label=f"Judge: {resp.evaluation.recommendation} (overall={resp.evaluation.overall_score}, grade={resp.evaluation.grade})",
        detail=resp.evaluation.overall_assessment,
        data={
            "evaluation_id": resp.metadata.evaluation_id,
            "accuracy": resp.evaluation.factual_accuracy.score,
            "completeness": resp.evaluation.completeness.score,
            "flags": resp.evaluation.flags,
        },
    )

    state.trace.append(f"Judge complete: recommendation={resp.evaluation.recommendation} overall={resp.evaluation.overall_score}")

    return state
