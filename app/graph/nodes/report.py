from __future__ import annotations

from app.graph.models import FinalReport, SearchResult
from app.graph.state import ResearchState


def _collect_sources(state: ResearchState, cap: int = 25) -> list[SearchResult]:
    sources: list[SearchResult] = []
    seen: set[str] = set()
    for s in state.searches:
        for r in s.results:
            url = str(r.url)
            if url in seen:
                continue
            seen.add(url)
            sources.append(r)
            if len(sources) >= cap:
                return sources
    return sources


async def report_node(state: ResearchState) -> ResearchState:
    sources = _collect_sources(state)

    # Prefer reasoning notes; otherwise derive minimal findings from top snippets.
    key_findings: list[str] = []
    if state.reasoning_notes:
        key_findings.extend(state.reasoning_notes[-12:])

    if not key_findings:
        for r in sources[:8]:
            snip = (r.snippet or "").strip()
            if not snip:
                continue
            # Keep findings short, snippet-derived (no hallucinations)
            key_findings.append(snip[:200] + ("…" if len(snip) > 200 else ""))
        key_findings = key_findings[:8]

    evidence: list[str] = []
    for r in sources[:12]:
        snippet = (r.snippet or "").strip()
        if snippet:
            evidence.append(f"{r.title} — {snippet} ({r.url})")
        else:
            evidence.append(f"{r.title} ({r.url})")

    # Do NOT include fetched page excerpts in the report output (too noisy / boilerplate-heavy).
    # We keep them only for internal reasoning/debugging.

    limitations: list[str] = []
    if state.iteration_count >= state.max_revisions:
        limitations.append(
            f"Max iterations reached ({state.max_revisions}); report is best-effort."
        )
    if not sources:
        limitations.append("No search results were collected (Serper errors or all queries rejected).")
    limitations.append("This agent may fetch linked pages (best-effort) to extract additional text; failures are logged.")

    state.report = FinalReport(
        key_findings=key_findings or ["No findings could be extracted from the collected evidence."],
        evidence_and_sources=evidence or ["No sources available."],
        limitations=limitations,
    )
    state.trace.append("Report generated")
    return state
