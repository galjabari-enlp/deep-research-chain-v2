from __future__ import annotations

import logging
from typing import List

from app.core import settings
from app.graph.enforce import dedupe_preserve_order, plan_all_topics, query_contains_any_topic
from app.graph.models import SearchQueryRecord
from app.graph.state import ResearchState
from app.services.serper import SerperClient, SerperError

logger = logging.getLogger(__name__)


def _candidate_queries(state: ResearchState) -> List[str]:
    if state.plan is None:
        return []

    # Priority: critic-requested gaps -> reasoning proposed -> plan initial batch
    gap_queries: List[str] = []
    if state.critic and state.critic.missing_gaps:
        gap_queries = [g.suggested_query for g in state.critic.missing_gaps]

    proposed = list(state.proposed_queries)
    initial = list(state.plan.initial_search_batch)

    return dedupe_preserve_order(gap_queries + proposed + initial)


def _validate_query_against_plan(query: str, state: ResearchState) -> tuple[bool, str]:
    if state.plan is None:
        return False, "No plan available"
    topics = plan_all_topics(state.plan)
    if not query_contains_any_topic(query, topics):
        return False, "Query does not match any plan topics"
    if state.critic and state.critic.constraints_for_next_search:
        # Ensure at least one constraint keyword appears, loosely.
        constraints = state.critic.constraints_for_next_search
        if constraints and not any(c.lower() in query.lower() for c in constraints):
            return False, "Query does not satisfy critic constraints"
    return True, "OK"


async def search_node(state: ResearchState, serper: SerperClient) -> ResearchState:
    if state.plan is None:
        state.trace.append("Search skipped: no plan")
        return state

    cap = max(1, settings.per_iteration_search_cap)
    candidates = _candidate_queries(state)

    executed_this_iter = 0
    existing_urls = {str(r.url) for s in state.searches for r in s.results}

    for q in candidates:
        if executed_this_iter >= cap:
            break
        if q.lower() in {x.lower() for x in state.search_queries}:
            continue

        ok, reason = _validate_query_against_plan(q, state)
        if not ok:
            state.trace.append(f"Rejected query '{q}': {reason}")
            continue

        record = SearchQueryRecord(query=q, rationale=f"Validated vs plan topics: {reason}")

        try:
            resp = await serper.search(q, top_n=settings.serper_top_n)
            # Deduplicate URLs across iterations
            for res in resp.results:
                if str(res.url) in existing_urls:
                    continue
                existing_urls.add(str(res.url))
                record.results.append(res)

            state.search_queries.append(q)
            state.searches.append(record)
            executed_this_iter += 1
            state.trace.append(f"Searched: {q} ({len(record.results)} new results)")
        except SerperError as e:
            logger.warning("Serper search failed for '%s': %s", q, e)
            state.trace.append(f"Serper error for '{q}': {e}")

    if executed_this_iter == 0:
        state.trace.append("No searches executed this iteration (all rejected or already done).")

    return state
