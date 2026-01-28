from __future__ import annotations

import asyncio
import json
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, conint, constr

# Mounting StaticFiles at a path that overlaps with API routes can be subtle.
# We'll mount the frontend at the end of the file so explicit API routes win.

from app.core import setup_logging
from app.graph import ResearchState, build_research_graph
from app.graph.models import ResearchResponse
from app.services import LLMClient, PageFetcher, SerperClient

logger = logging.getLogger(__name__)
logger.info("UI static dirs: dist=%s assets=%s", "frontend/dist", "frontend/dist/assets")

# Explicitly load `.env` for API runs.
load_dotenv(override=False)

setup_logging("INFO")

app = FastAPI(title="Deep Research Agent", version="0.1.0")


class ResearchRequest(BaseModel):
    query: constr(min_length=3, max_length=500)
    max_revisions: conint(ge=1, le=10) = Field(default=10)


# UI index (React build)
@app.get("/")
async def ui_index() -> FileResponse:
    return FileResponse("frontend/dist/index.html")


@app.get("/ui")
async def ui_index_slashless() -> FileResponse:
    return FileResponse("frontend/dist/index.html")


# NOTE: Keep a separate legacy UI route during migration for easy rollback/comparison.
@app.get("/legacy")
async def legacy_ui() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/config")
async def debug_config() -> dict[str, object]:
    """Debug-only: return non-sensitive config details to verify runtime env.

    This intentionally avoids returning full secrets.
    """

    from app.core import settings

    key = settings.openai_api_key or ""
    return {
        "openai_model": settings.openai_model,
        "openai_base_url": settings.openai_base_url or None,
        "openai_api_key_len": len(key),
        "openai_api_key_tail": key[-4:] if len(key) >= 4 else "",
        "serper_api_key_len": len(settings.serper_api_key or ""),
    }


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    llm = LLMClient()
    serper = SerperClient()
    fetcher = PageFetcher()
    graph = build_research_graph(llm=llm, serper=serper, fetcher=fetcher)

    init = ResearchState(query=req.query, max_revisions=req.max_revisions)
    raw_state = await graph.ainvoke(init, config={"recursion_limit": 200})

    # LangGraph may return either the dataclass state or an internal AddableValuesDict.
    final_state = ResearchState(**raw_state) if isinstance(raw_state, dict) else raw_state

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
        execution_trace=final_state.execution_trace,
        critic=final_state.critic,
    )


@app.get("/research/stream")
async def research_stream(query: constr(min_length=3, max_length=500), max_revisions: conint(ge=1, le=10) = 10):
    """Server-Sent Events stream.

    Emits JSON messages:
    - {type: 'state', stage, iteration_count, trace, critic, ts_ms}
    - {type: 'final', response: ResearchResponse}
    - {type: 'error', message}

    This allows the UI to display live stage + iteration progress while the graph runs.
    """

    llm = LLMClient()
    serper = SerperClient()
    fetcher = PageFetcher()
    graph = build_research_graph(llm=llm, serper=serper, fetcher=fetcher)

    async def gen():
        try:
            ts_ms = lambda: int(asyncio.get_event_loop().time() * 1000)

            init_state = ResearchState(query=query, max_revisions=max_revisions)

            # Initial event
            yield f"data: {json.dumps({'type': 'state', 'stage': 'planning', 'iteration_count': 0, 'trace': [], 'critic': None, 'ts_ms': ts_ms()})}\n\n"

            last_stage = None
            last_iter = None
            last_trace_len = None

            # IMPORTANT: consume astream_events on the same run; do not run ainvoke() separately until the stream finishes.
            final_state_obj = None
            async for event in graph.astream_events(
                init_state,
                version="v2",
                config={"recursion_limit": 200},
            ):
                if event.get("event") != "on_chain_end":
                    continue

                name = (event.get("name") or "").strip()
                if name in {"planning", "search", "reasoning", "critic_step", "report_step"}:
                    raw_out = event.get("data", {}).get("output")
                    if raw_out is None:
                        continue

                    state = ResearchState(**raw_out) if isinstance(raw_out, dict) else raw_out

                    stage = name
                    iter_count = getattr(state, "iteration_count", None)
                    trace = getattr(state, "trace", None)
                    trace_len = len(trace) if isinstance(trace, list) else None

                    critic = getattr(state, "critic", None)
                    critic_dump = critic.model_dump(mode="json") if critic is not None else None

                    if stage != last_stage or iter_count != last_iter or trace_len != last_trace_len:
                        last_stage, last_iter, last_trace_len = stage, iter_count, trace_len
                        payload = {
                            "type": "state",
                            "stage": stage,
                            "iteration_count": iter_count,
                            "trace": trace if isinstance(trace, list) else [],
                            "execution_trace": state.execution_trace.model_dump(mode="json"),
                            "critic": critic_dump,
                            "ts_ms": ts_ms(),
                            "plan": state.plan.model_dump(mode="json") if state.plan is not None else None,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

                # Capture final state from the same run
                if event.get("event") == "on_chain_end" and (event.get("name") or "").strip() == "LangGraph":
                    final_state_obj = event.get("data", {}).get("output")

            if final_state_obj is None:
                # Fallback: if no final captured, invoke once.
                raw_state = await graph.ainvoke(init_state, config={"recursion_limit": 200})
                final_state = ResearchState(**raw_state) if isinstance(raw_state, dict) else raw_state
            else:
                final_state = ResearchState(**final_state_obj) if isinstance(final_state_obj, dict) else final_state_obj

            sources = []
            for s in final_state.searches:
                sources.extend(s.results)

            report = final_state.report
            if report is None:
                from app.graph.models import FinalReport

                report = FinalReport(
                    key_findings=["No report generated due to internal error."],
                    evidence_and_sources=["No sources available."],
                    limitations=["Internal error prevented report generation."],
                )

            response = ResearchResponse(
                query=query,
                plan=final_state.plan,
                iteration_count=final_state.iteration_count,
                report=report,
                sources=sources,
                trace=final_state.trace,
                execution_trace=final_state.execution_trace,
                critic=final_state.critic,
            )

            # Debug: heading completeness for UI blocks
            try:
                blocks = (report.blocks or []) if report is not None else []
                empty = sum(1 for b in blocks if not (getattr(b, "heading", "") or "").strip())
                logger.info(
                    "Final report blocks: count=%s empty_heading=%s",
                    len(blocks),
                    empty,
                )
                if empty:
                    sample = [
                        {
                            "id": getattr(b, "id", ""),
                            "heading": getattr(b, "heading", ""),
                            "text_preview": (getattr(b, "text", "") or "")[:120],
                        }
                        for b in blocks[:5]
                    ]
                    logger.warning("Empty headings detected in final report blocks. sample=%s", sample)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to log report block headings: %s", e)

            yield f"data: {json.dumps({'type': 'final', 'response': response.model_dump(mode='json')})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---- Frontend mount (place last so explicit API routes win) ----
# Mount under /ui so it doesn't intercept API endpoints like /health, /research/*.
# NOTE: assets are served from /assets by the mount below.
app.mount("/ui", StaticFiles(directory="frontend/dist", html=True), name="frontend")

# Serve built assets at the root (/assets/...) to match Vite's default build output.
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
