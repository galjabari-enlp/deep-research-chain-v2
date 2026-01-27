from __future__ import annotations

from typing import Any

import pytest

from app.graph.build_graph import build_research_graph
from app.graph.models import CriticAssessment, FinalReport, ResearchPlan
from app.graph.state import ResearchState


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def chat_json(self, *, system: str, user: str, **_: Any) -> str:
        self.calls.append((system, user))
        if "ResearchPlan" in user:
            # minimal valid plan JSON (must satisfy Pydantic constraints)
            return (
                "{"
                '"plan_version": 1,'
                '"user_query": "What is X?",'
                '"research_objective": "Understand the concept of X and its implications using credible sources.",'
                '"scope_inclusions": [],'
                '"scope_exclusions": [],'
                '"key_terms": ["concept","definition","impacts"],'
                '"sub_questions": ["What is X (definitions and common usage)?","What are the key impacts or implications of X?"],'
                '"source_type_requirements": [{"source_type":"reputable_news","rationale":"Need a baseline overview from reputable reporting to corroborate definitions and impacts.","must_have": true}],'
                '"steps": ['
                '{"step_id":"S1","objective":"Define X clearly using authoritative summaries.","topics":["definition"],"must_find":[],"suggested_queries":["X definition","X overview definition"],"preferred_source_types":[],"freshness_need":"any"},'
                '{"step_id":"S2","objective":"Identify impacts or implications of X described by sources.","topics":["impacts"],"must_find":[],"suggested_queries":["X impacts","X implications impacts"],"preferred_source_types":[],"freshness_need":"any"}'
                '],'
                '"success_criteria": ['
                '{"criterion_id":"C1","description":"At least 2 credible sources explain X using snippet evidence.","must_be_met": true},'
                '{"criterion_id":"C2","description":"At least 1 credible source mentions impacts/implications of X.","must_be_met": true}'
                '],'
                '"initial_search_batch": ["X definition","X overview definition","X impacts"],'
                '"notes_for_search_agent": ""'
                "}"
            )
        # Critic JSON: immediately report
        return (
            "{"
            '"iteration": 1,'
            '"decision": "report",'
            '"sufficiency_score": 80,'
            '"what_we_have": ["We have snippets."],'
            '"evidence_quality": [],'
            '"unmet_success_criteria": [],'
            '"missing_gaps": [],'
            '"plan_issues": [],'
            '"feedback_to_planning": "",'
            '"constraints_for_next_search": []'
            "}"
        )


class FakeSerper:
    async def search(self, query: str, *, top_n: int | None = None):
        from app.services.serper import SerperSearchResponse
        from app.graph.models import SearchResult

        return SerperSearchResponse(
            results=[
                SearchResult(
                    title=f"Result for {query}",
                    url="https://example.com",
                    snippet=f"Snippet for {query}",
                )
            ],
            raw={},
        )


@pytest.mark.asyncio
async def test_graph_runs_to_report() -> None:
    llm = FakeLLM()
    serper = FakeSerper()
    graph = build_research_graph(llm=llm, serper=serper)

    init = ResearchState(query="What is X?", max_revisions=3)
    out = await graph.ainvoke(init)
    if isinstance(out, dict):
        # With dataclass state, LangGraph returns a dict of state keys -> values.
        final_state = ResearchState(**out)
    else:
        final_state = out

    assert isinstance(final_state.plan, ResearchPlan)
    assert isinstance(final_state.critic, CriticAssessment)
    assert final_state.critic.decision == "report"
    assert isinstance(final_state.report, FinalReport)
    assert final_state.searches
