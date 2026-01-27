from __future__ import annotations

import logging

from app.graph.enforce import (
    model_validate_json_with_retry,
    plan_all_topics,
    sanitize_for_prompt,
    validate_critic_constraints,
)
from app.graph.models import CriticAssessment
from app.graph.state import ResearchState
from app.services.llm import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a strict research critic. You must ensure the agent does not stray from the plan. "
    "Be extremely concise and obey field length limits. "
    "Hard limits: MissingGap.description <= 220 chars; EvidenceQuality.reasons entries <= 180 chars; "
    "PlanIssue fields <= 220 chars; feedback_to_planning <= 600 chars. "
    "Do not include unrelated content. Use short phrases, not sentences, unless required. "
    "Return ONLY valid JSON matching the provided schema. No markdown, no commentary."
)


def _critic_user_prompt(state: ResearchState) -> str:
    assert state.plan is not None
    schema = CriticAssessment.model_json_schema()

    sources = []
    for s in state.searches[-6:]:
        for r in s.results[:5]:
            sources.append({"title": r.title, "url": str(r.url), "snippet": r.snippet})

    return (
        f"Iteration: {state.iteration_count}\n"
        f"User query: {sanitize_for_prompt(state.query)}\n"
        f"Plan topics (do not stray): {plan_all_topics(state.plan)}\n"
        f"Success criteria: {[c.criterion_id + ': ' + c.description for c in state.plan.success_criteria]}\n\n"
        f"What we have (recent sources/snippets): {sources}\n\n"
        "Assess sufficiency, evidence quality, and gaps vs success criteria.\n\n"
        "IMPORTANT OUTPUT RULES (must follow):\n"
        "- Return ONLY JSON. No extra keys.\n"
        "- Be concise: each string must be short.\n"
        "- missing_gaps[*].description MUST be <= 220 characters (aim <= 140).\n"
        "- Use 1-3 short reasons per evidence item.\n"
        "- Do NOT include unrelated topics/content.\n\n"
        "Choose decision:\n"
        "- report: if sufficient (score >=75, no unmet criteria, no non-minor gaps)\n"
        "- refine_search: if gaps remain; provide missing_gaps with concrete suggested_query strings\n"
        "- replan: if plan is flawed; provide plan_issues and feedback_to_planning\n\n"
        "Return ONLY valid JSON matching this schema (JSON Schema follows).\n"
        f"SCHEMA: {schema}"
    )


async def critic_node(state: ResearchState, llm: LLMClient) -> ResearchState:
    if state.plan is None:
        state.trace.append("Critic skipped: no plan")
        return state

    state.iteration_count += 1

    user_prompt = _critic_user_prompt(state)
    raw = await llm.chat_json(system=SYSTEM_PROMPT, user=user_prompt)

    critic, err = model_validate_json_with_retry(CriticAssessment, raw)

    if critic is None:
        retry = (
            user_prompt
            + "\n\nYour previous output failed validation. "
            + f"Validation error: {sanitize_for_prompt(err or 'unknown', 1200)}\n"
            + "Return ONLY valid JSON matching the schema."
        )
        raw2 = await llm.chat_json(system=SYSTEM_PROMPT, user=retry)
        critic, err2 = model_validate_json_with_retry(CriticAssessment, raw2)
        if critic is None:
            state.trace.append("Critic failed: unable to produce valid JSON; forcing report.")
            return state

    assert isinstance(critic, CriticAssessment)
    critic.iteration = min(max(1, state.iteration_count), state.max_revisions)

    errs = validate_critic_constraints(critic)
    if errs:
        # Re-ask once with constraint errors.
        retry = (
            user_prompt
            + "\n\nYour JSON validated structurally, but violated decision constraints:\n"
            + "\n".join(f"- {e}" for e in errs)
            + "\nFix and return ONLY valid JSON."
        )
        raw3 = await llm.chat_json(system=SYSTEM_PROMPT, user=retry)
        critic2, _ = model_validate_json_with_retry(CriticAssessment, raw3)
        if isinstance(critic2, CriticAssessment):
            critic = critic2
        else:
            # Force downgrade decision
            critic.decision = "refine_search"
            critic.sufficiency_score = min(critic.sufficiency_score, 70)

    state.critic = critic
    state.critic_history.append(critic)
    state.trace.append(f"Critic decision: {critic.decision} (score={critic.sufficiency_score})")
    logger.info("Critic decision=%s score=%s", critic.decision, critic.sufficiency_score)
    return state
