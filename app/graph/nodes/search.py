from __future__ import annotations

import logging
from typing import List

from app.core import settings
from app.graph.enforce import dedupe_preserve_order, plan_all_topics, query_contains_any_topic
from app.graph.models import SearchQueryRecord
from app.graph.state import ResearchState
from app.services.fetcher import FetchError, PageFetcher, extract_text_from_html
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
        # If the critic provides constraints that are too strict (leading to zero executable queries),
        # allow fallback to plan-topic validation only.
        constraints = [c.strip() for c in state.critic.constraints_for_next_search if c.strip()]
        if constraints and not any(c.lower() in query.lower() for c in constraints):
            return False, "Query does not satisfy critic constraints"
    return True, "OK"


async def search_node(
    state: ResearchState, serper: SerperClient, fetcher: PageFetcher | None = None
) -> ResearchState:
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

            from app.graph.trace_models import add_trace_item, make_link

            add_trace_item(
                state.execution_trace,
                iteration=state.iteration_count,
                section="search_queries",
                label=q,
                data={"new_results": len(record.results)},
            )

            if record.results:
                add_trace_item(
                    state.execution_trace,
                    iteration=state.iteration_count,
                    section="search_results",
                    label=f"Results for: {q}",
                    links=[make_link(title=res.title, url=str(res.url)) for res in record.results[:10]],
                )

            for res in record.results:
                state.trace.append(f"Source considered: {res.title} | {res.url}")

            # Optional page fetch: actually visit URLs and extract text for downstream reasoning.
            if fetcher is not None:
                for res in record.results:
                    url = str(res.url)
                    if url in state.fetched_pages:
                        continue
                    try:
                        fr = await fetcher.fetch(url)
                        state.trace.append(
                            f"Fetched: {url} (status={fr.status_code}, content_type={fr.content_type})"
                        )

                        from app.graph.trace_models import add_trace_item

                        add_trace_item(
                            state.execution_trace,
                            iteration=state.iteration_count,
                            section="fetch_details",
                            label="Fetched",
                            detail=str(url),
                            data={
                                "url": str(url),
                                "status_code": fr.status_code,
                                "content_type": fr.content_type,
                            },
                        )

                        if fr.status_code < 400 and (fr.content_type or "").lower().startswith("text/"):
                            state.fetched_pages[url] = extract_text_from_html(fr.text)
                        else:
                            state.fetched_pages[url] = ""
                    except FetchError as e:
                        state.trace.append(f"Fetch error: {url} ({e})")

                        from app.graph.trace_models import add_trace_item

                        add_trace_item(
                            state.execution_trace,
                            iteration=state.iteration_count,
                            section="fetch_details",
                            label="Fetch error",
                            detail=str(url),
                            data={"url": str(url), "error": str(e)},
                        )
        except SerperError as e:
            logger.warning("Serper search failed for '%s': %s", q, e)
            state.trace.append(f"Serper error for '{q}': {e}")

    if executed_this_iter == 0:
        # Fallback: if *all* candidates were rejected due to critic constraints, rerun allowing plan-only validation.
        rejected_due_to_constraints = any(
            "critic constraints" in t.lower() for t in state.trace[-25:]
        )
        if rejected_due_to_constraints:
            state.trace.append("All candidates blocked by critic constraints; retrying with plan-only validation.")
            for q in candidates:
                if executed_this_iter >= cap:
                    break
                if q.lower() in {x.lower() for x in state.search_queries}:
                    continue

                # Temporarily ignore critic constraints
                saved = state.critic.constraints_for_next_search if state.critic else None
                if state.critic:
                    state.critic.constraints_for_next_search = []
                ok, reason = _validate_query_against_plan(q, state)
                if state.critic and saved is not None:
                    state.critic.constraints_for_next_search = saved

                if not ok:
                    continue

                record = SearchQueryRecord(query=q, rationale=f"Validated vs plan topics (constraint fallback): {reason}")
                try:
                    resp = await serper.search(q, top_n=settings.serper_top_n)
                    for res in resp.results:
                        if str(res.url) in existing_urls:
                            continue
                        existing_urls.add(str(res.url))
                        record.results.append(res)

                    state.search_queries.append(q)
                    state.searches.append(record)
                    executed_this_iter += 1
                    state.trace.append(f"Searched (constraint fallback): {q} ({len(record.results)} new results)")
                except SerperError as e:
                    state.trace.append(f"Serper error for '{q}' (constraint fallback): {e}")

        if executed_this_iter == 0:
            state.trace.append("No searches executed this iteration (all rejected or already done).")

    return state
