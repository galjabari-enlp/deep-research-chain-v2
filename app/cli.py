from __future__ import annotations

import argparse
import asyncio
import json

from dotenv import load_dotenv

from app.core import setup_logging
from app.graph import ResearchState, build_research_graph
from app.services import LLMClient, SerperClient


def _print_report(state: ResearchState) -> None:
    print("\n=== TRACE ===")
    for t in state.trace:
        print(f"- {t}")

    if state.plan:
        print("\n=== PLAN (objective) ===")
        print(state.plan.research_objective)
        print("\nInitial search batch:")
        for q in state.plan.initial_search_batch:
            print(f"- {q}")

    print("\n=== SEARCHES EXECUTED ===")
    for s in state.searches:
        print(f"- {s.query} ({len(s.results)} results)")

    if state.critic:
        print("\n=== FINAL CRITIC ===")
        print(json.dumps(state.critic.model_dump(), indent=2))

    print("\n=== REPORT ===")
    if state.report:
        print("\nKey findings:")
        for b in state.report.key_findings:
            print(f"- {b}")
        print("\nEvidence & sources:")
        for b in state.report.evidence_and_sources:
            print(f"- {b}")
        print("\nLimitations:")
        for b in state.report.limitations:
            print(f"- {b}")
    else:
        print("(no report generated)")


async def _run(query: str, max_revisions: int) -> int:
    llm = LLMClient()
    serper = SerperClient()
    graph = build_research_graph(llm=llm, serper=serper)

    init = ResearchState(query=query, max_revisions=max_revisions)
    out = await graph.ainvoke(init)

    # LangGraph returns a dict of state keys -> values when using dataclass state.
    final_state: ResearchState = ResearchState(**out) if isinstance(out, dict) else out

    _print_report(final_state)
    return 0


def main() -> int:
    # Explicitly load `.env` for CLI runs.
    load_dotenv(override=False)

    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Deep Research Agent CLI")
    parser.add_argument("query", type=str, help="Research question")
    parser.add_argument("--max-revisions", type=int, default=10)
    args = parser.parse_args()

    return asyncio.run(_run(args.query, args.max_revisions))


if __name__ == "__main__":
    raise SystemExit(main())
