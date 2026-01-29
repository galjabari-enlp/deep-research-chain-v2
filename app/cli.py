from __future__ import annotations

import argparse
import asyncio
import json

from dotenv import load_dotenv

from app.cli_render import render_evaluation_card, render_missing_evaluation_card
from app.core import setup_logging
from app.graph import ResearchState, build_research_graph
from app.services import LLMClient, PageFetcher, SerperClient


def _print_report(state: ResearchState, *, verbose_trace: bool, show_urls: bool) -> None:
    # CLI can't truly collapse, but we can approximate:
    # - default: print only non-collapsed sections (plan + critic + report)
    # - verbose: print all sections

    et = getattr(state, "execution_trace", None)

    if et is not None and getattr(et, "iterations", None):
        print("\n=== EXECUTION TRACE (sectioned) ===")
        for it in et.iterations:
            print(f"\n--- Iteration {it.iteration} ---")
            for sec in it.sections:
                if (not verbose_trace) and sec.collapsed_by_default:
                    continue
                print(f"\n[{sec.title}]")
                for item in sec.items:
                    line = f"- {item.label}"
                    if item.detail:
                        line += f" | {item.detail}"
                    print(line)
                    for link in item.links:
                        dom = f" ({link.domain})" if link.domain else ""
                        print(f"  - [{link.title}]{dom}")
                        if show_urls:
                            print(f"    {link.url}")
    else:
        # Back-compat fallback
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

    # ---- Judge evaluation card ----
    if getattr(state, "public_report", None) is not None and getattr(state, "judge_evaluation", None) is not None:
        print("\n=== REPORT EVALUATION ===")
        print(render_evaluation_card(report=state.public_report, evaluation=state.judge_evaluation))
    elif getattr(state, "public_report", None) is not None:
        print("\n=== REPORT EVALUATION ===")
        print(render_missing_evaluation_card(report=state.public_report))


async def _run(query: str, max_revisions: int, *, verbose_trace: bool, show_urls: bool) -> int:
    llm = LLMClient()
    serper = SerperClient()
    fetcher = PageFetcher()
    graph = build_research_graph(llm=llm, serper=serper, fetcher=fetcher)

    init = ResearchState(query=query, max_revisions=max_revisions)
    out = await graph.ainvoke(init)

    # LangGraph returns a dict of state keys -> values when using dataclass state.
    final_state: ResearchState = ResearchState(**out) if isinstance(out, dict) else out

    _print_report(final_state, verbose_trace=verbose_trace, show_urls=show_urls)
    return 0


def main() -> int:
    # Explicitly load `.env` for CLI runs.
    load_dotenv(override=False)

    # Debug: confirm settings resolved from environment.
    from app.core import settings

    key = settings.openai_api_key or ""
    print(
        f"[debug] settings.openai_model={settings.openai_model!r} base_url={settings.openai_base_url!r} "
        f"api_key_len={len(key)} api_key_tail={key[-4:] if len(key) >= 4 else ''!r}"
    )

    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Deep Research Agent CLI")
    parser.add_argument("query", type=str, help="Research question")
    parser.add_argument("--max-revisions", type=int, default=10)
    parser.add_argument(
        "--verbose-trace",
        action="store_true",
        help="Print all trace sections (including collapsed-by-default ones).",
    )
    parser.add_argument(
        "--show-urls",
        action="store_true",
        help="Print full URLs under each link entry.",
    )
    args = parser.parse_args()

    return asyncio.run(
        _run(
            args.query,
            args.max_revisions,
            verbose_trace=bool(args.verbose_trace),
            show_urls=bool(args.show_urls),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
