from __future__ import annotations

from typing import Any

import pytest

from app.graph.build_graph import build_research_graph
from app.graph.models import CriticAssessment, FinalReport, ResearchPlan
from app.graph.state import ResearchState


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.plan_calls = 0
        self.critic_calls = 0

    async def chat_json(self, *, system: str, user: str, **_: Any) -> str:
        self.calls.append((system, user))
        # planning node includes the model schema in the user prompt; match on that.
        if "SCHEMA" in user and "ResearchPlan" in user:
            # (this block returns plan JSON)
            self.plan_calls += 1
            pv = self.plan_calls
            # minimal valid plan JSON (must satisfy Pydantic constraints)
            return (
                "{"
                f'"plan_version": {pv},'
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
                '"notes_for_search_agent": "avoid repeating these queries: X definition"'
                "}"
            )

        # Critic JSON: refine_search on first iteration, then report on second.
        # Detect critic calls by presence of "CriticAssessment" schema in the prompt.
        if "SCHEMA" in user and "CriticAssessment" in user:
            self.critic_calls += 1
            if self.critic_calls == 1:
                return (
                    "{"
                    '"iteration": 1,'
                    '"decision": "refine_search",'
                    '"sufficiency_score": 60,'
                    '"what_we_have": ["We have some snippets."],'
                    '"evidence_quality": [],'
                    '"unmet_success_criteria": ["C2"],'
                    '"missing_gaps": ['
                    '{"gap_id":"G1","description":"Need evidence about impacts/implications.","mapped_to_success_criteria":["C2"],"severity":"important","suggested_query":"X impacts","required_source_types":[]}'
                    '],'
                    '"plan_issues": [],'
                    '"feedback_to_planning": "Incorporate impacts gaps into steps and queries.",'
                    '"constraints_for_next_search": []'
                    "}"
                )

            return (
                "{"
                '"iteration": 2,'
                '"decision": "report",'
                '"sufficiency_score": 80,'
                '"what_we_have": ["We have enough snippets."],'
                '"evidence_quality": [],'
                '"unmet_success_criteria": [],'
                '"missing_gaps": [],'
                '"plan_issues": [],'
                '"feedback_to_planning": "",'
                '"constraints_for_next_search": []'
                "}"
            )

        # Reasoning node expects plain text, but our graph uses llm.chat_json for it.
        # Return a minimal text with the required sections.
        return "KNOWN:\n- We have snippets\nGAPS:\n- impacts evidence missing\nQUERIES:\n- X impacts\n"


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

    # New behavior: if critic is not satisfied (refine_search), we still replan.
    assert len(final_state.plan_history) >= 2
    assert final_state.plan_history[0].plan_version == 1
    assert final_state.plan_history[1].plan_version == 2

    # Ensure planning happened again before the second search.
    # The trace should contain a second "Planned v2".
    assert any(t.startswith("Planned v2") for t in final_state.trace)

    # Structured execution_trace shape + collapsed defaults
    et = final_state.execution_trace
    assert et.iterations
    for it in et.iterations:
        sections = {s.section: s for s in it.sections}
        # required sections
        assert "plan" in sections
        assert "search_queries" in sections
        assert "search_results" in sections
        assert "critic" in sections

        assert sections["plan"].collapsed_by_default is False
        assert sections["critic"].collapsed_by_default is False
        assert sections["search_queries"].collapsed_by_default is True
        assert sections["search_results"].collapsed_by_default is True
        assert sections["fetch_details"].collapsed_by_default is True
        assert sections["reasoning"].collapsed_by_default is True
        assert sections["report"].collapsed_by_default is False
