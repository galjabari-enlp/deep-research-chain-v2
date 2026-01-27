from __future__ import annotations

import logging

from app.graph.enforce import dedupe_preserve_order
from app.graph.state import ResearchState
from app.services.llm import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a careful research assistant. Use ONLY the provided search snippets; "
    "do not invent facts. Produce concise notes and targeted follow-up queries."
)


def _snippets_context(state: ResearchState, max_items: int = 25) -> str:
    items = []
    for s in state.searches:
        for r in s.results:
            items.append(f"- {r.title} | {r.url} | {r.snippet}")
    if len(items) > max_items:
        items = items[:max_items]
    return "\n".join(items)


async def reasoning_node(state: ResearchState, llm: LLMClient) -> ResearchState:
    if state.plan is None:
        state.trace.append("Reasoning skipped: no plan")
        return state

    context = _snippets_context(state)
    user = (
        f"User query: {state.query}\n"
        f"Success criteria: {[c.criterion_id + ': ' + c.description for c in state.plan.success_criteria]}\n\n"
        "Search snippets (may be empty):\n"
        f"{context or '- (none)'}\n\n"
        "Task:\n"
        "1) Summarize what we know so far using ONLY snippet evidence (no extra facts).\n"
        "2) Identify gaps vs success criteria.\n"
        "3) Propose 2-6 follow-up search queries (strings) tightly scoped to plan topics.\n\n"
        "Return as plain text with sections:\n"
        "KNOWN:\n- ...\nGAPS:\n- ...\nQUERIES:\n- ...\n"
    )

    try:
        text = await llm.chat_json(system=SYSTEM_PROMPT, user=user)
    except Exception as e:  # noqa: BLE001
        logger.warning("Reasoning LLM failed: %s", e)
        state.reasoning_notes.append("Reasoning step failed due to LLM error.")
        return state

    known: list[str] = []
    gaps: list[str] = []
    queries: list[str] = []
    section = None
    for line in (text or "").splitlines():
        l = line.strip()
        if not l:
            continue
        upper = l.rstrip(":").upper()
        if upper in {"KNOWN", "GAPS", "QUERIES"}:
            section = upper
            continue
        if l.startswith("-"):
            item = l.lstrip("-").strip()
            if section == "KNOWN":
                known.append(item)
            elif section == "GAPS":
                gaps.append(item)
            elif section == "QUERIES":
                queries.append(item)

    state.reasoning_notes.extend(known)
    state.gaps = dedupe_preserve_order(state.gaps + gaps)
    state.proposed_queries = dedupe_preserve_order(queries)
    state.trace.append(f"Reasoning produced {len(known)} notes, {len(gaps)} gaps, {len(queries)} queries")
    return state
