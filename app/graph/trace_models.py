from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

TraceSectionType = Literal[
    "plan",
    "search_queries",
    "search_results",
    "fetch_details",
    "reasoning",
    "critic",
    "report",
]


class TraceLink(BaseModel):
    title: str
    url: str
    domain: Optional[str] = None


class TraceItem(BaseModel):
    ts: Optional[str] = None
    label: str
    detail: Optional[str] = None
    links: List[TraceLink] = Field(default_factory=list)
    data: Optional[Dict[str, Any]] = None


class TraceSection(BaseModel):
    section: TraceSectionType
    title: str
    collapsed_by_default: bool = True
    items: List[TraceItem] = Field(default_factory=list)


class IterationTrace(BaseModel):
    iteration: int
    sections: List[TraceSection] = Field(default_factory=list)


class ExecutionTrace(BaseModel):
    iterations: List[IterationTrace] = Field(default_factory=list)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _domain_from_url(url: str) -> str | None:
    if not url:
        return None
    try:
        if "://" in url:
            return url.split("/")[2]
    except Exception:
        return None
    return None


def _default_sections() -> list[TraceSection]:
    # MUST match collapsed defaults from spec.
    return [
        TraceSection(section="plan", title="Plans", collapsed_by_default=False),
        TraceSection(section="search_queries", title="Search queries executed", collapsed_by_default=True),
        TraceSection(section="search_results", title="Sources considered / results", collapsed_by_default=True),
        TraceSection(section="fetch_details", title="Fetch/HTTP details", collapsed_by_default=True),
        TraceSection(section="critic", title="Critic decisions", collapsed_by_default=False),
        TraceSection(section="reasoning", title="Reasoning summaries", collapsed_by_default=True),
        TraceSection(section="report", title="Report", collapsed_by_default=False),
    ]


def _ensure_iteration(trace: ExecutionTrace, iteration: int) -> IterationTrace:
    for it in trace.iterations:
        if it.iteration == iteration:
            return it
    it = IterationTrace(iteration=iteration, sections=_default_sections())
    trace.iterations.append(it)
    trace.iterations.sort(key=lambda x: x.iteration)
    return it


def _ensure_section(it: IterationTrace, section: TraceSectionType) -> TraceSection:
    for s in it.sections:
        if s.section == section:
            return s
    # Shouldn't happen (we pre-create), but safe fallback.
    sec = TraceSection(section=section, title=section, collapsed_by_default=True)
    it.sections.append(sec)
    return sec


def add_trace_item(
    trace: ExecutionTrace,
    *,
    iteration: int,
    section: TraceSectionType,
    label: str,
    detail: str | None = None,
    links: list[TraceLink] | None = None,
    data: dict[str, Any] | None = None,
    ts: str | None = None,
) -> None:
    it = _ensure_iteration(trace, iteration)
    sec = _ensure_section(it, section)
    sec.items.append(
        TraceItem(
            ts=ts or _utc_ts(),
            label=label,
            detail=detail,
            links=links or [],
            data=data,
        )
    )


def make_link(*, title: str, url: str) -> TraceLink:
    return TraceLink(title=title, url=url, domain=_domain_from_url(url))
