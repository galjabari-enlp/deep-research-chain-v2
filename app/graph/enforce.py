from __future__ import annotations

import json
import logging
import re
from typing import Iterable, List, Sequence

from pydantic import ValidationError

from .models import CriticAssessment, MissingGap, ResearchPlan

logger = logging.getLogger(__name__)


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        k = item.strip()
        if not k:
            continue
        low = k.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(k)
    return out


def query_contains_any_topic(query: str, topics: Sequence[str]) -> bool:
    q = query.lower()
    return any(t.lower() in q for t in topics if t.strip())


def plan_all_topics(plan: ResearchPlan) -> List[str]:
    topics: List[str] = []
    for step in plan.steps:
        topics.extend(step.topics)
    return dedupe_preserve_order(topics)


def validate_and_fix_plan(plan: ResearchPlan) -> ResearchPlan:
    """Enforce project-specific rules beyond Pydantic constraints."""

    # 1) initial_search_batch must be a deduped union of step suggested_queries, capped 3–8.
    union: List[str] = []
    for step in plan.steps:
        union.extend(step.suggested_queries)
    union = dedupe_preserve_order(union)
    union = union[: max(3, min(8, len(union)))]
    plan.initial_search_batch = union

    # 2) suggested_queries must contain at least one keyword from step topics.
    for step in plan.steps:
        fixed: List[str] = []
        for q in step.suggested_queries:
            if query_contains_any_topic(q, step.topics):
                fixed.append(q)
            else:
                # programmatic fix: append the first topic in quotes
                topic = step.topics[0]
                fixed.append(f"{q} \"{topic}\"")
        step.suggested_queries = fixed

    return plan


def validate_gap_query_against_plan(gap: MissingGap, plan: ResearchPlan) -> bool:
    topics = plan_all_topics(plan)
    return query_contains_any_topic(gap.suggested_query, topics)


def validate_critic_constraints(critic: CriticAssessment) -> List[str]:
    errs: List[str] = []
    if critic.decision == "report":
        if critic.sufficiency_score < 75:
            errs.append("decision=report requires sufficiency_score >= 75")
        non_minor = [g for g in critic.missing_gaps if g.severity != "minor"]
        if non_minor:
            errs.append("decision=report requires missing_gaps to be empty or only minor")
        if critic.unmet_success_criteria:
            errs.append("decision=report requires unmet_success_criteria to be empty")

    if critic.decision == "replan":
        if not critic.plan_issues:
            errs.append("decision=replan requires plan_issues non-empty")
        if not critic.feedback_to_planning.strip():
            errs.append("decision=replan requires feedback_to_planning non-empty")

    if critic.decision == "refine_search":
        if not critic.missing_gaps:
            errs.append("decision=refine_search requires missing_gaps non-empty")
        for g in critic.missing_gaps:
            if not g.suggested_query.strip():
                errs.append("missing_gaps must include suggested_query")
    return errs


def safe_parse_json_object(text: str) -> str:
    """Attempt to extract a JSON object from text; returns raw object string."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    # naive extraction: first '{' to last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def model_validate_json_with_retry(model_cls, llm_text: str) -> tuple[object | None, str | None]:
    """Single attempt parse helper. Returns (model|None, error|None)."""
    raw = safe_parse_json_object(llm_text)
    try:
        return model_cls.model_validate_json(raw), None
    except ValidationError as e:
        logger.warning("Validation failed for %s: %s", model_cls.__name__, e)
        return None, str(e)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed for %s: %s", model_cls.__name__, e)
        return None, str(e)


def sanitize_for_prompt(text: str, max_len: int = 2000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text
