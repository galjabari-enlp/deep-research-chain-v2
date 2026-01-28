from __future__ import annotations

import logging

from app.graph.enforce import model_validate_json_with_retry, sanitize_for_prompt, validate_and_fix_plan
from app.graph.models import ResearchPlan
from app.graph.state import ResearchState
from app.services.llm import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a senior research planner. You create realistic, high-quality web search plans. "
    "You MUST match the user's language in ALL text fields and queries. "
    "If the user writes in Bulgarian, all steps/topics/queries MUST be Bulgarian. "
    "Return ONLY valid JSON matching the provided schema. No markdown, no commentary."
)


def _planning_user_prompt(
    *,
    query: str,
    plan_version: int,
    feedback: str,
    previous_plan_json: str = "",
    critic_json: str = "",
    evidence_summary: str = "",
    executed_queries: list[str] | None = None,
) -> str:
    schema = ResearchPlan.model_json_schema()
    executed_queries = executed_queries or []

    prev_block = previous_plan_json.strip() or "(none)"
    critic_block = critic_json.strip() or "(none)"
    evidence_block = evidence_summary.strip() or "(none)"

    return (
        f"User query: {sanitize_for_prompt(query)}\n"
        f"Plan version (must increment): {plan_version}\n\n"
        "Inputs you must use when replanning:\n"
        f"1) Previous plan JSON (may be none on v1): {sanitize_for_prompt(prev_block, 2400)}\n\n"
        f"2) Latest critic assessment JSON (may be none on v1): {sanitize_for_prompt(critic_block, 2400)}\n\n"
        f"3) Evidence summary (titles/domains + snippet summary): {sanitize_for_prompt(evidence_block, 2400)}\n\n"
        f"4) Search queries already executed (avoid repeating): {executed_queries}\n\n"
        f"Critic feedback to incorporate (may be empty): {sanitize_for_prompt(feedback, 800)}\n\n"
        "Task: Create an UPDATED ResearchPlan JSON matching the schema exactly.\n"
        "- Preserve what worked; adjust what didn’t based on critic + evidence.\n"
        "- Keep aligned to the original user query.\n"
        "- Update steps[*].topics and steps[*].suggested_queries as needed.\n"
        "- Keep success_criteria stable unless critic indicates they are unrealistic.\n"
        "- notes_for_search_agent MUST include constraints like:\n"
        "  - avoid repeating these queries: ...\n"
        "  - prioritize missing gaps: ...\n"
        "  - prefer source types: ...\n"
        "LANGUAGE RULE: All fields MUST be in the same language as the user query.\n"
        "Important rules:\n"
        "- steps[*].suggested_queries must each include at least one keyword from steps[*].topics (substring, case-insensitive).\n"
        "- initial_search_batch must be a deduplicated union of all steps[*].suggested_queries, capped to 3-8 items.\n"
        "- Keep queries short and actionable; avoid unrelated topics; avoid repeating executed queries.\n\n"
        f"SCHEMA (JSON Schema): {schema}"
    )


async def planning_node(state: ResearchState, llm: LLMClient) -> ResearchState:
    feedback = ""
    if state.critic and state.critic.feedback_to_planning:
        feedback = state.critic.feedback_to_planning

    plan_version = 1
    if state.plan_history:
        plan_version = max(p.plan_version for p in state.plan_history) + 1

    # Planning inputs required for replanning
    prev_plan_json = state.plan.model_dump_json() if state.plan is not None else ""
    critic_json = state.critic.model_dump_json() if state.critic is not None else ""

    # compact evidence summary: titles + domains + snippets (recent)
    evidence_items: list[str] = []
    for s in state.searches[-6:]:
        for r in s.results[:5]:
            url = str(r.url)
            domain = url.split("/")[2] if "://" in url and len(url.split("/")) >= 3 else url
            snip = (r.snippet or "").strip().replace("\n", " ")
            if len(snip) > 220:
                snip = snip[:220] + "…"
            evidence_items.append(f"- {domain} | {r.title} | {snip}")
    evidence_summary = "\n".join(evidence_items[:25])

    executed_queries = list(state.search_queries)

    user_prompt = _planning_user_prompt(
        query=state.query,
        plan_version=plan_version,
        feedback=feedback,
        previous_plan_json=prev_plan_json,
        critic_json=critic_json,
        evidence_summary=evidence_summary,
        executed_queries=executed_queries,
    )

    raw = await llm.chat_json(system=SYSTEM_PROMPT, user=user_prompt)
    plan, err = model_validate_json_with_retry(ResearchPlan, raw)

    if plan is None:
        # Re-ask once with validation errors.
        retry_prompt = (
            user_prompt
            + "\n\nYour previous output failed validation. "
            + f"Validation error: {sanitize_for_prompt(err or 'unknown', 1200)}\n"
            + "Return ONLY valid JSON matching the schema."
        )
        raw2 = await llm.chat_json(system=SYSTEM_PROMPT, user=retry_prompt)
        plan, err2 = model_validate_json_with_retry(ResearchPlan, raw2)
        if plan is None:
            state.trace.append("Planning failed: unable to produce valid JSON plan.")
            state.reasoning_notes.append(
                "Planning failed to validate; proceeding with best-effort report."
            )
            return state

    assert isinstance(plan, ResearchPlan)
    plan.plan_version = plan_version
    plan.user_query = state.query
    plan = validate_and_fix_plan(plan)

    state.plan = plan
    state.plan_history.append(plan)

    # Record which plan_version will be used after the current critic iteration.
    # If no critic has run yet, iteration_count may be 0; store under 0.
    state.iteration_plan_versions[state.iteration_count] = plan.plan_version

    from app.graph.trace_models import add_trace_item

    add_trace_item(
        state.execution_trace,
        iteration=state.iteration_count,
        section="plan",
        label=f"Plan v{plan.plan_version}",
        detail=plan.research_objective,
        data={
            "plan_version": plan.plan_version,
            "initial_search_batch": list(plan.initial_search_batch),
        },
    )

    state.trace.append(f"Planned v{plan.plan_version}: {plan.research_objective}")
    logger.info("Plan created (v%s) with %s initial queries", plan.plan_version, len(plan.initial_search_batch))
    logger.debug("Plan schema: %s", ResearchPlan.model_json_schema())
    return state
