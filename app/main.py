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
from app.graph.judge_models import JudgeResponse
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


class ResearchReviseRequest(BaseModel):
    # Stateless revision: frontend sends prior context back.
    query: constr(min_length=3, max_length=500)
    user_note: constr(min_length=0, max_length=1200) = ""

    # Prior public report + judge evaluation to drive revision.
    report: dict
    evaluation: dict
    metadata: dict | None = None

    # Prior sources (optional). If present, we’ll reuse them.
    sources: list[dict] = Field(default_factory=list)


class PublishRequest(BaseModel):
    query: constr(min_length=3, max_length=500)

    # Full run payload (client-posted). This endpoint persists artifacts and does not re-run the graph.
    report: dict
    evaluation: dict | None = None
    metadata: dict | None = None
    sources: list[dict] = Field(default_factory=list)


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


@app.post("/research", response_model=JudgeResponse)
async def research(req: ResearchRequest) -> JudgeResponse:
    llm = LLMClient()
    serper = SerperClient()
    fetcher = PageFetcher()
    graph = build_research_graph(llm=llm, serper=serper, fetcher=fetcher)

    init = ResearchState(query=req.query, max_revisions=req.max_revisions)
    raw_state = await graph.ainvoke(init, config={"recursion_limit": 200})

    # LangGraph may return either the dataclass state or an internal AddableValuesDict.
    final_state = ResearchState(**raw_state) if isinstance(raw_state, dict) else raw_state

    # Ensure report + evaluation exist (best-effort fallbacks)
    if final_state.public_report is None:
        # Fallback: reuse legacy report mapping
        from app.graph.judge_models import PublicReport

        final_state.public_report = PublicReport(
            id="rep_error",
            topic=req.query,
            content="",
            sources=[],
            word_count=0,
            created_at=None,
        )

    if final_state.judge_evaluation is None:
        # Judge may fail validation; return a safe placeholder evaluation.
        from app.graph.judge_models import JudgeEvaluation

        final_state.judge_evaluation = JudgeEvaluation(
            factual_accuracy={
                "score": 0,
                "max_score": 10,
                "percentage": 0,
                "reasoning": "Judge failed to evaluate due to internal error.",
                "strengths": [],
                "weaknesses": ["Evaluation unavailable"],
            },
            completeness={
                "score": 0,
                "max_score": 10,
                "percentage": 0,
                "reasoning": "Judge failed to evaluate due to internal error.",
                "strengths": [],
                "weaknesses": ["Evaluation unavailable"],
                "coverage": {},
            },
            overall_score=0,
            grade="F",
            overall_assessment="Evaluation could not be produced due to an internal error.",
            recommendation="revise",
            confidence=0.0,
            flags=["judge-failed"],
            suggested_improvements=["Re-run evaluation"],
        )

    if final_state.judge_metadata is None:
        from app.graph.judge_models import JudgeMetadata

        final_state.judge_metadata = JudgeMetadata(
            evaluation_id=None,
            evaluated_at=None,
            judge_model="claude-sonnet-4-20250514",
            processing_time_ms=None,
            evaluation_version="1.0",
        )

    return JudgeResponse(
        status="complete",
        report=final_state.public_report,
        evaluation=final_state.judge_evaluation,
        metadata=final_state.judge_metadata,
    )


@app.post("/api/reports/{report_id}/revise", response_model=JudgeResponse)
async def report_revise(report_id: str, req: ResearchReviseRequest) -> JudgeResponse:
    """Regenerate report using judge feedback + optional user note.

    Stateless: client POSTs prior `report` + `evaluation` (and optionally `sources`).

    Versioning:
    - creates a new report id: <old_id>_v<next>
    - does not overwrite previous report
    """

    from app.graph.judge_models import JudgeEvaluation, JudgeMetadata, PublicReport

    llm = LLMClient()
    serper = SerperClient()
    fetcher = PageFetcher()
    graph = build_research_graph(llm=llm, serper=serper, fetcher=fetcher)

    # Parse prior evaluation/report defensively.
    base_eval = JudgeEvaluation.model_validate(req.evaluation)
    prior_sources = list(req.sources or [])

    init = ResearchState(query=req.query, max_revisions=1)
    init.revision_base_evaluation = base_eval
    init.revision_user_note = (req.user_note or "").strip()

    # Carry a versioned report id forward.
    base_report_id = str((req.report or {}).get("id") or report_id).strip() or report_id
    # If base already has _vN, bump it; else set v2.
    import re as _re

    m = _re.match(r"^(?P<root>.+)_v(?P<n>\d+)$", base_report_id)
    if m:
        root = m.group("root")
        n = int(m.group("n"))
        next_id = f"{root}_v{n + 1}"
    else:
        next_id = f"{base_report_id}_v2"

    try:
        init.public_report = PublicReport.model_validate({**(req.report or {}), "id": next_id})
    except Exception:
        init.public_report = PublicReport(id=next_id, topic=req.query, content="", sources=[], word_count=0, created_at=None)

    if prior_sources and init.public_report is not None:
        init.public_report.sources = prior_sources

    raw_state = await graph.ainvoke(init, config={"recursion_limit": 120})
    final_state = ResearchState(**raw_state) if isinstance(raw_state, dict) else raw_state

    if final_state.public_report is None:
        final_state.public_report = PublicReport(id="rep_error", topic=req.query, content="", sources=prior_sources, word_count=0, created_at=None)

    if final_state.judge_evaluation is None:
        final_state.judge_evaluation = base_eval

    if final_state.judge_metadata is None:
        final_state.judge_metadata = JudgeMetadata(
            evaluation_id=None,
            evaluated_at=None,
            judge_model="claude-sonnet-4-20250514",
            processing_time_ms=None,
            evaluation_version="1.0",
        )

    # Ensure the response carries the versioned id even if internal nodes rebuilt the report.
    if final_state.public_report is not None:
        final_state.public_report.id = next_id

    return JudgeResponse(status="complete", report=final_state.public_report, evaluation=final_state.judge_evaluation, metadata=final_state.judge_metadata)


@app.post("/api/reports/{report_id}/publish")
async def publish_report(report_id: str, req: PublishRequest) -> dict:
    """Publish (persist) a report artifact to disk.

    Rules enforced server-side:
    - evaluation.recommendation must be "publish"
    - factual_accuracy.score >= 7
    - completeness.score >= 7

    Storage format (preferred): write two files under ./reports:
    - <id>__<published_at>.md (report content + sources)
    - <id>__<published_at>.json (full payload: report+evaluation+metadata+published_at)
    """

    import json as _json
    import os
    from datetime import datetime, timezone

    from fastapi import HTTPException

    base_dir = "reports"
    os.makedirs(base_dir, exist_ok=True)

    report_obj = req.report if isinstance(req.report, dict) else {}
    eval_obj = req.evaluation if isinstance(req.evaluation, dict) else None
    meta_obj = req.metadata if isinstance(req.metadata, dict) else None

    # Enforce publish gating.
    recommendation = (eval_obj or {}).get("recommendation")
    factual_score = ((eval_obj or {}).get("factual_accuracy") or {}).get("score")
    completeness_score = ((eval_obj or {}).get("completeness") or {}).get("score")

    def _to_int(v):
        try:
            return int(v)
        except Exception:
            return None

    factual_i = _to_int(factual_score)
    complete_i = _to_int(completeness_score)

    if recommendation != "publish" or factual_i is None or complete_i is None or factual_i < 7 or complete_i < 7:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Publish not allowed by Judge gating rules.",
                "required": {
                    "recommendation": "publish",
                    "factual_accuracy_min": 7,
                    "completeness_min": 7,
                },
                "actual": {
                    "recommendation": recommendation,
                    "factual_accuracy_score": factual_i,
                    "completeness_score": complete_i,
                },
            },
        )

    # Use request report.id if present; otherwise fall back to URL id.
    report_id_final = str(report_obj.get("id") or report_id).strip() or report_id

    published_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    stamp = published_at.replace(":", "-")
    base_name = f"{report_id_final}__{stamp}"

    md_path = os.path.join(base_dir, f"{base_name}.md")
    json_path = os.path.join(base_dir, f"{base_name}.json")

    # Write markdown report content
    topic = str(report_obj.get("topic") or req.query or "").strip()
    content = str(report_obj.get("content") or "").strip()
    blocks = report_obj.get("blocks") if isinstance(report_obj.get("blocks"), list) else None

    md_lines: list[str] = []
    if topic:
        md_lines.append(f"# {topic}")
        md_lines.append("")

    if isinstance(blocks, list) and blocks:
        for b in blocks:
            if not isinstance(b, dict):
                continue
            heading = str(b.get("heading") or "").strip()
            text = str(b.get("text") or "").strip()
            if heading:
                md_lines.append(f"## {heading}")
            if text:
                md_lines.append(text)
            md_lines.append("")
    elif content:
        md_lines.append(content)
        md_lines.append("")

    sources = req.sources if isinstance(req.sources, list) else []
    sources = [s for s in sources if isinstance(s, dict)]

    # Always include Sources section.
    md_lines.append("## Sources")
    if sources:
        for s in sources:
            title = str(s.get("title") or "").strip()
            url = str(s.get("url") or "").strip()
            if not url:
                continue
            label = title or url
            md_lines.append(f"- {label} ({url})")
    else:
        md_lines.append("- (none)")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines).strip() + "\n")

    payload = {
        "report": {
            "id": report_id_final,
            "topic": topic,
            "content": content,
            "sources": report_obj.get("sources") if isinstance(report_obj.get("sources"), list) else [],
            "word_count": report_obj.get("word_count"),
            "created_at": report_obj.get("created_at"),
        },
        "evaluation": eval_obj,
        "metadata": meta_obj,
        "query": req.query,
        "sources": sources,
        "published_at": published_at,
        "artifacts": {
            "md": md_path,
            "json": json_path,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"ok": True, "report_id": report_id_final, "published_at": published_at, "artifacts": {"md": md_path, "json": json_path}}


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
        """SSE generator.

        IMPORTANT:
        - We must yield early and periodically; otherwise browsers may close the connection.
        - Do not run `astream_events()` twice; it would run the graph twice.
        """

        try:
            ts_ms = lambda: int(asyncio.get_event_loop().time() * 1000)

            init_state = ResearchState(query=query, max_revisions=max_revisions)

            # Initial event (send ASAP so browsers don't treat the stream as dead/hung).
            yield f"data: {json.dumps({'type': 'state', 'stage': 'planning', 'iteration_count': 0, 'trace': [], 'critic': None, 'ts_ms': ts_ms()})}\n\n"

            # Force a flush for servers/proxies that buffer small chunks.
            yield ": init\n" + (" " * 2048) + "\n\n"

            last_stage = None
            last_iter = None
            last_trace_len = None

            final_state_obj = None

            # Heartbeat interval
            heartbeat_every_s = 10
            next_heartbeat_at = asyncio.get_event_loop().time() + heartbeat_every_s

            async for event in graph.astream_events(
                init_state,
                version="v2",
                config={"recursion_limit": 200},
            ):
                now = asyncio.get_event_loop().time()
                if now >= next_heartbeat_at:
                    yield f": keep-alive {ts_ms()}\n\n"
                    next_heartbeat_at = now + heartbeat_every_s

                if event.get("event") != "on_chain_end":
                    continue

                name = (event.get("name") or "").strip()
                if name not in {"planning", "search", "reasoning", "critic_step", "report_step", "judge_step"}:
                    continue

                raw_out = event.get("data", {}).get("output")
                if raw_out is None:
                    continue

                state = ResearchState(**raw_out) if isinstance(raw_out, dict) else raw_out

                stage = name
                iter_count = getattr(state, "iteration_count", None)
                trace = getattr(state, "trace", None)
                trace_len = len(trace) if isinstance(trace, list) else None

                # Critic is a Pydantic model (CriticAssessment) and is NOT JSON-serializable by default.
                critic_obj = getattr(state, "critic", None)
                critic = critic_obj.model_dump(mode="json") if critic_obj is not None else None

                # Only emit when something meaningful changes.
                if stage == last_stage and iter_count == last_iter and trace_len == last_trace_len:
                    continue

                last_stage = stage
                last_iter = iter_count
                last_trace_len = trace_len

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "state",
                            "stage": stage,
                            "iteration_count": iter_count,
                            "trace": trace or [],
                            "critic": critic,
                            "ts_ms": ts_ms(),
                        }
                    )
                    + "\n\n"
                )

                final_state_obj = state

            if final_state_obj is not None:
                # Build a JSON-safe final payload.
                # NOTE: `ResearchResponse` currently has no `from_state()` helper; stream the state-derived fields directly.
                try:
                    report_obj = getattr(final_state_obj, "report", None)
                    plan_obj = getattr(final_state_obj, "plan", None)
                    critic_obj = getattr(final_state_obj, "critic", None)
                    exec_trace_obj = getattr(final_state_obj, "execution_trace", None)

                    response = {
                        "query": getattr(final_state_obj, "query", query),
                        "plan": plan_obj.model_dump(mode="json") if plan_obj is not None else None,
                        "iteration_count": getattr(final_state_obj, "iteration_count", None),
                        "report": report_obj.model_dump(mode="json") if report_obj is not None else {"blocks": []},
                        "sources": [
                            r.model_dump(mode="json")
                            for s in (getattr(final_state_obj, "searches", None) or [])
                            for r in (getattr(s, "results", None) or [])
                        ],
                        "trace": getattr(final_state_obj, "trace", None) or [],
                        "execution_trace": (
                            exec_trace_obj.model_dump(mode="json") if exec_trace_obj is not None else None
                        ),
                        "critic": critic_obj.model_dump(mode="json") if critic_obj is not None else None,
                    }
                except Exception as _e:
                    # Last-resort: don’t fail the whole SSE stream on final payload shaping.
                    response = {
                        "query": getattr(final_state_obj, "query", query),
                        "iteration_count": getattr(final_state_obj, "iteration_count", None),
                        "trace": getattr(final_state_obj, "trace", None) or [],
                    }

                yield f"data: {json.dumps({'type': 'final', 'response': response})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

    return StreamingResponse(gen(), media_type="text/event-stream")
