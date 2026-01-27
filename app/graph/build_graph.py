from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from app.graph.enforce import validate_gap_query_against_plan
from app.graph.nodes.critic import critic_node
from app.graph.nodes.planning import planning_node
from app.graph.nodes.reasoning import reasoning_node
from app.graph.nodes.report import report_node
from app.graph.nodes.search import search_node
from app.graph.state import ResearchState
from app.services.llm import LLMClient
from app.services.serper import SerperClient


def _route_from_critic(state: ResearchState) -> Literal["planning", "search", "report"]:
    # Max-iterations hard stop.
    if state.iteration_count >= state.max_revisions:
        return "report"

    if state.critic is None:
        return "report"

    decision = state.critic.decision

    if decision == "report":
        return "report"

    if decision == "replan":
        return "planning"

    # refine_search
    if state.plan is None:
        return "planning"

    # Validate suggested queries against plan topics; if invalid, replan.
    if state.critic.missing_gaps:
        for gap in state.critic.missing_gaps:
            if not validate_gap_query_against_plan(gap, state.plan):
                # Critic asked for an off-plan query => plan/critic misaligned.
                return "planning"

    return "search"


def build_research_graph(*, llm: LLMClient, serper: SerperClient):
    graph = StateGraph(ResearchState)

    async def planning(state: ResearchState) -> ResearchState:
        return await planning_node(state, llm)

    async def search(state: ResearchState) -> ResearchState:
        return await search_node(state, serper)

    async def reasoning(state: ResearchState) -> ResearchState:
        return await reasoning_node(state, llm)

    async def critic_step(state: ResearchState) -> ResearchState:
        return await critic_node(state, llm)

    async def report_step(state: ResearchState) -> ResearchState:
        return await report_node(state)

    graph.add_node("planning", planning)
    graph.add_node("search", search)
    graph.add_node("reasoning", reasoning)
    # Node names must not collide with state keys (ResearchState has a 'critic' field)
    graph.add_node("critic_step", critic_step)
    # Node names must not collide with state keys (ResearchState has a 'report' field)
    graph.add_node("report_step", report_step)

    graph.set_entry_point("planning")

    graph.add_edge("planning", "search")
    graph.add_edge("search", "reasoning")
    graph.add_edge("reasoning", "critic_step")

    graph.add_conditional_edges(
        "critic_step",
        _route_from_critic,
        {
            "planning": "planning",
            "search": "search",
            "report": "report_step",
        },
    )

    graph.add_edge("report_step", END)

    return graph.compile()
