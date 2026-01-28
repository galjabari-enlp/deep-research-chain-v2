from __future__ import annotations

import logging

from app.graph.models import FinalReport, SearchResult
from app.graph.state import ResearchState

logger = logging.getLogger(__name__)


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
    """Generate the final report.

    Preferred: ask the LLM to write natural-prose paragraphs and explicit citations per paragraph.
    Fallback: build blocks from existing snippet-derived findings.
    """

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

    # New UI contract
    from app.graph.models import ReportBlock, ReportCitation

    blocks: list[ReportBlock] = []

    # 1) Preferred: LLM writes blocks in prose (no '**Label**:' field prefixes)
    try:
        from app.services.llm import LLMClient

        llm = LLMClient()

        system = (
            "You are a careful research writer. Write polished, natural prose. "
            "Do NOT use bold label prefixes like '**X**:' or dictionary-entry field labels. "
            "Write complete sentences and paragraphs. "
            "Only cite from the provided sources."
        )

        source_lines: list[str] = []
        for idx, r in enumerate(sources[:12], start=1):
            snip = (r.snippet or "").strip().replace("\n", " ")
            if len(snip) > 280:
                snip = snip[:280] + "…"
            source_lines.append(f"S{idx}. {r.title} | {r.url} | {snip}")

        user = (
            f"User query: {state.query}\n\n"
            "Sources (use these only):\n"
            + "\n".join(source_lines)
            + "\n\n"
            "Task: produce 3-8 blocks. Each block must have a short heading and one paragraph of prose. "
            "After each paragraph, include 1-4 citations referencing the sources by id (S1..S12).\n"
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            "  \"blocks\": [\n"
            "    {\"id\": \"B1\", \"heading\": \"...\", \"text\": \"...paragraph...\", \"citations\": [\n"
            "      {\"id\": \"S1\", \"url\": \"https://...\", \"title\": \"...\"}\n"
            "    ]}\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Each block must include 'heading' (2-6 words) for the sub-section title; no markdown, no trailing colon.\n"
            "- 'text' must be plain prose without leading labels like '**Something**:'\n"
            "- Do not use markdown headings.\n"
            "- Each citation id must correspond to one of the provided sources.\n"
        )

        raw = await llm.chat_json(system=system, user=user)
        import json as _json

        obj = _json.loads(raw)
        raw_blocks = obj.get("blocks") if isinstance(obj, dict) else None
        if isinstance(raw_blocks, list) and raw_blocks:
            for b in raw_blocks[:40]:
                if not isinstance(b, dict):
                    continue
                text = str(b.get("text") or "").strip()
                if not text:
                    continue
                heading = str(b.get("heading") or "").strip()

                cits_in = b.get("citations")
                citations: list[ReportCitation] = []
                if isinstance(cits_in, list):
                    for c in cits_in[:12]:
                        if not isinstance(c, dict):
                            continue
                        cid = str(c.get("id") or "").strip()
                        url = c.get("url")
                        title = str(c.get("title") or "")
                        if cid and url:
                            citations.append(ReportCitation(id=cid, url=url, title=title))

                blocks.append(
                    ReportBlock(
                        id=str(b.get("id") or ""),
                        heading=heading,
                        text=text,
                        citations=citations,
                    )
                )

    except Exception as e:  # noqa: BLE001
        logger.info("LLM report blocks generation failed; falling back to snippet-derived blocks: %s", e)

    # 2) Fallback: build blocks from key findings + broad citations
    if not blocks:
        for i, txt in enumerate(key_findings[:12]):
            txt = (txt or "").strip()
            if not txt:
                continue

            citations: list[ReportCitation] = []
            for j, r in enumerate(sources[:4]):
                citations.append(ReportCitation(id=f"S{j + 1}", url=r.url, title=r.title or ""))

            blocks.append(ReportBlock(id=f"B{i + 1}", heading="", text=txt, citations=citations))

    # Legacy evidence list remains (but UI uses blocks primarily)
    evidence: list[str] = []
    for r in sources[:12]:
        snippet = (r.snippet or "").strip()
        if snippet:
            evidence.append(f"{r.title} — {snippet} ({r.url})")
        else:
            evidence.append(f"{r.title} ({r.url})")

    limitations: list[str] = []
    if state.iteration_count >= state.max_revisions:
        limitations.append(f"Max iterations reached ({state.max_revisions}); report is best-effort.")
    if not sources:
        limitations.append("No search results were collected (Serper errors or all queries rejected).")
    limitations.append("This agent may fetch linked pages (best-effort) to extract additional text; failures are logged.")

    state.report = FinalReport(
        key_findings=key_findings or ["No findings could be extracted from the collected evidence."],
        evidence_and_sources=evidence or ["No sources available."],
        limitations=limitations,
        blocks=blocks,
    )

    from app.graph.trace_models import add_trace_item, make_link

    add_trace_item(
        state.execution_trace,
        iteration=state.iteration_count,
        section="report",
        label="Report generated",
        detail=("\n".join(state.report.key_findings[:5]) if state.report else None),
        links=[make_link(title=r.title, url=str(r.url)) for r in sources[:10]],
    )

    state.trace.append("Report generated")
    return state
