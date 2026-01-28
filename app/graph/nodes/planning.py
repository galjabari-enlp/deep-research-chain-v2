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


def _planning_user_prompt(query: str, plan_version: int, feedback: str) -> str:
    schema = ResearchPlan.model_json_schema()
    return (
        f"User query: {sanitize_for_prompt(query)}\n"
        f"Plan version: {plan_version}\n"
        f"Critic feedback to incorporate (may be empty): {sanitize_for_prompt(feedback, 800)}\n\n"
        "Create a ResearchPlan JSON matching this schema exactly.\n"
        "LANGUAGE RULE: All fields MUST be in the same language as the user query.\n"
        "Important rules:\n"
        "- steps[*].suggested_queries must each include at least one keyword from steps[*].topics (substring, case-insensitive).\n"
        "- initial_search_batch must be a deduplicated union of all steps[*].suggested_queries, capped to 3-8 items.\n"
        "- Keep queries short and actionable, avoid adding unrelated topics.\n\n"
        f"SCHEMA (JSON Schema): {schema}"
    )


async def planning_node(state: ResearchState, llm: LLMClient) -> ResearchState:
    feedback = ""
    if state.critic and state.critic.feedback_to_planning:
        feedback = state.critic.feedback_to_planning

    plan_version = 1
    if state.plan_history:
        plan_version = max(p.plan_version for p in state.plan_history) + 1

    user_prompt = _planning_user_prompt(state.query, plan_version, feedback)

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
    state.trace.append(f"Planned v{plan.plan_version}: {plan.research_objective}")
    logger.info("Plan created (v%s) with %s initial queries", plan.plan_version, len(plan.initial_search_batch))
    logger.debug("Plan schema: %s", ResearchPlan.model_json_schema())
    return state
