from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, constr

from app.core import setup_logging
from app.graph import ResearchState, build_research_graph
from app.graph.models import ResearchResponse
from app.services import LLMClient, SerperClient

logger = logging.getLogger(__name__)

# Explicitly load `.env` for API runs.
load_dotenv(override=False)

setup_logging("INFO")

app = FastAPI(title="Deep Research Agent", version="0.1.0")


class ResearchRequest(BaseModel):
    query: constr(min_length=3, max_length=500)
    max_revisions: conint(ge=1, le=10) = Field(default=10)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    llm = LLMClient()
    serper = SerperClient()
    graph = build_research_graph(llm=llm, serper=serper)

    init = ResearchState(query=req.query, max_revisions=req.max_revisions)
    final_state: ResearchState = await graph.ainvoke(init)

    sources = []
    for s in final_state.searches:
        sources.extend(s.results)

    report = final_state.report
    if report is None:
        # Fallback best-effort
        from app.graph.models import FinalReport

        report = FinalReport(
            key_findings=["No report generated due to internal error."],
            evidence_and_sources=["No sources available."],
            limitations=["Internal error prevented report generation."],
        )

    return ResearchResponse(
        query=req.query,
        plan=final_state.plan,
        iteration_count=final_state.iteration_count,
        report=report,
        sources=sources,
        trace=final_state.trace,
    )
