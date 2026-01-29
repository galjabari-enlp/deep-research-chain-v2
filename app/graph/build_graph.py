from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from app.graph.enforce import validate_gap_query_against_plan
from app.graph.nodes.critic import critic_node
from app.graph.nodes.judge import judge_node
from app.graph.nodes.planning import planning_node
from app.graph.nodes.policy_guard import policy_guard_node
from app.graph.nodes.reasoning import reasoning_node
from app.graph.nodes.report import report_node
from app.graph.nodes.search import search_node
from app.graph.state import ResearchState
from app.services.fetcher import PageFetcher
from app.services.llm import LLMClient
from app.services.serper import SerperClient


def _route_from_critic(state: ResearchState) -> Literal["planning", "report"]:
    """Routing policy: replan after *every* non-satisfactory critic evaluation.

    Terminal conditions:
    - max_revisions reached -> report
    - critic.decision == "report" -> report

    Otherwise always route to planning, regardless of whether critic chose
    "refine_search" or "replan".
    """

    # Max-iterations hard stop.
    if state.iteration_count >= state.max_revisions:
        state.trace.append("Routing: max_revisions reached -> report")
        return "report"

    if state.critic is None:
        state.trace.append("Routing: missing critic -> report")
        return "report"

    decision = state.critic.decision

    if decision == "report":
        state.trace.append("Routing: critic=report -> report")
        return "report"

    state.trace.append(f"Routing: critic={decision} -> planning (always replan when not reporting)")
    return "planning"


def build_research_graph(*, llm: LLMClient, serper: SerperClient, fetcher: PageFetcher | None = None):
    graph = StateGraph(ResearchState)

    async def policy_guard(state: ResearchState) -> ResearchState:
        return await policy_guard_node(state, llm)

    async def planning(state: ResearchState) -> ResearchState:
        return await planning_node(state, llm)

    async def search(state: ResearchState) -> ResearchState:
        return await search_node(state, serper, fetcher)

    async def reasoning(state: ResearchState) -> ResearchState:
        return await reasoning_node(state, llm)

    async def critic_step(state: ResearchState) -> ResearchState:
        return await critic_node(state, llm)

    async def report_step(state: ResearchState) -> ResearchState:
        return await report_node(state)

    async def judge_step(state: ResearchState) -> ResearchState:
        return await judge_node(state, llm)

    graph.add_node("policy_guard", policy_guard)
    graph.add_node("planning", planning)
    graph.add_node("search", search)
    graph.add_node("reasoning", reasoning)
    # Node names must not collide with state keys (ResearchState has a 'critic' field)
    graph.add_node("critic_step", critic_step)
    # Node names must not collide with state keys (ResearchState has a 'report' field)
    graph.add_node("report_step", report_step)
    graph.add_node("judge_step", judge_step)

    graph.set_entry_point("policy_guard")

    def _route_from_guard(state: ResearchState) -> Literal["planning", "end"]:
        if getattr(state, "status", "") == "blocked":
            state.trace.append("Routing: policy blocked -> END")
            return "end"
        return "planning"

    graph.add_conditional_edges(
        "policy_guard",
        _route_from_guard,
        {
            "planning": "planning",
            "end": END,
        },
    )

    graph.add_edge("planning", "search")
    graph.add_edge("search", "reasoning")
    graph.add_edge("reasoning", "critic_step")

    graph.add_conditional_edges(
        "critic_step",
        _route_from_critic,
        {
            "planning": "planning",
            "report": "report_step",
        },
    )

    graph.add_edge("report_step", "judge_step")
    graph.add_edge("judge_step", END)

    return graph.compile()
