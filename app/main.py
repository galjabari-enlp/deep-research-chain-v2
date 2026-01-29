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

    # Force a single revision run (hard requirement for the UI button).
    max_revisions: conint(ge=1, le=1) = Field(default=1)

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

    import os

    from app.core import settings

    key = settings.openai_api_key or ""
    return {
        "openai_model": settings.openai_model,
        "openai_base_url": settings.openai_base_url or None,
        "openai_api_key_len": len(key),
        "openai_api_key_tail": key[-4:] if len(key) >= 4 else "",
        "serper_api_key_len": len(settings.serper_api_key or ""),
        "debug_allow_publish": str(os.environ.get("DEBUG_ALLOW_PUBLISH") or "").strip(),
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

        from app.core import settings

        final_state.judge_metadata = JudgeMetadata(
            evaluation_id=None,
            evaluated_at=None,
            judge_model=str(getattr(settings, "openai_model", "") or "").strip() or None,
            processing_time_ms=None,
            evaluation_version="1.0",
        )

    if getattr(final_state, "status", "") == "blocked":
        # Blocked runs intentionally return an empty report + a refusal message.
        from app.graph.judge_models import PublicReport

        # For blocked runs, return a minimal payload: no report/evaluation UI.
        return JudgeResponse(
            status="blocked",
            report=PublicReport(
                id="rep_blocked",
                topic=req.query,
                content=str(getattr(final_state, "final_user_message", "") or ""),
                sources=[],
                word_count=0,
                created_at=None,
            ),
            evaluation=None,
            metadata=final_state.judge_metadata,
        )

    return JudgeResponse(
        status="complete",
        report=final_state.public_report,
        evaluation=final_state.judge_evaluation,
        metadata=final_state.judge_metadata,
    )


@app.post("/api/reports/{report_id}/revise")
async def report_revise(report_id: str, req: ResearchReviseRequest):
    """Regenerate report using judge feedback + optional user note.

    Stateless: client POSTs prior `report` + `evaluation` (and optionally `sources`).

    Versioning:
    - creates a new report id: <old_id>_v<next>
    - does not overwrite previous report
    """

    from app.graph.judge_models import JudgeEvaluation, JudgeMetadata, PublicReport
    from fastapi import HTTPException
    import traceback

    llm = LLMClient()
    serper = SerperClient()

    # IMPORTANT: revisions must execute searches like a normal run.
    # Without a fetcher, reports often have no sources and the execution trace looks empty.
    fetcher = PageFetcher()
    graph = build_research_graph(llm=llm, serper=serper, fetcher=fetcher)

    # Parse prior evaluation/report defensively.
    # Frontend evaluation payloads may be "lite" (missing required fields like confidence/reasoning).
    # For revision, we only need the judge feedback fields used by planning_node (flags, suggested_improvements,
    # overall_assessment, recommendation). So accept partial input.
    try:
        base_eval = JudgeEvaluation.model_validate(req.evaluation)
    except Exception:
        # Best-effort partial normalization.
        ev = req.evaluation if isinstance(req.evaluation, dict) else {}
        fa = ev.get("factual_accuracy") if isinstance(ev.get("factual_accuracy"), dict) else {}
        co = ev.get("completeness") if isinstance(ev.get("completeness"), dict) else {}

        def _pct(score: int, max_score: int = 10) -> int:
            try:
                return int(round((int(score) / int(max_score)) * 100))
            except Exception:
                return 0

        fa_score = int(fa.get("score") or 0)
        co_score = int(co.get("score") or 0)
        base_eval = JudgeEvaluation.model_validate(
            {
                "factual_accuracy": {
                    "score": fa_score,
                    "max_score": int(fa.get("max_score") or 10),
                    "percentage": int(fa.get("percentage") or _pct(fa_score, int(fa.get("max_score") or 10))),
                    "reasoning": str(fa.get("reasoning") or "(missing)") or "(missing)",
                    "strengths": list(fa.get("strengths") or []),
                    "weaknesses": list(fa.get("weaknesses") or []),
                },
                "completeness": {
                    "score": co_score,
                    "max_score": int(co.get("max_score") or 10),
                    "percentage": int(co.get("percentage") or _pct(co_score, int(co.get("max_score") or 10))),
                    "reasoning": str(co.get("reasoning") or "(missing)") or "(missing)",
                    "strengths": list(co.get("strengths") or []),
                    "weaknesses": list(co.get("weaknesses") or []),
                    "coverage": co.get("coverage") if isinstance(co.get("coverage"), dict) else {},
                },
                "overall_score": float(ev.get("overall_score") or 0),
                "grade": str(ev.get("grade") or "F"),
                "overall_assessment": str(ev.get("overall_assessment") or "(missing assessment)").ljust(10, ".")[:2000],
                "recommendation": str(ev.get("recommendation") or "revise"),
                "confidence": float(ev.get("confidence") or 0.5),
                "flags": list(ev.get("flags") or []),
                "suggested_improvements": list(ev.get("suggested_improvements") or []),
            }
        )

    prior_sources = list(req.sources or [])

    init = ResearchState(query=req.query, max_revisions=1)
    init.revision_base_evaluation = base_eval
    init.revision_user_note = (req.user_note or "").strip()

    # Ensure revision uses prior context (reuse sources if provided; allow new searches as needed).
    # Seed the state with the prior public report + sources so planning/report can build on them.
    try:
        init.public_report = PublicReport.model_validate(req.report) if isinstance(req.report, dict) else None
    except Exception:
        init.public_report = None

    # Keep a copy of the prior report for planning/report prompts (what must change vs remain).
    init.revision_prior_report = init.public_report

    if prior_sources and init.public_report is not None:
        init.public_report.sources = prior_sources

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

    # For revision runs, reset execution artifacts so the response reflects THIS run only.
    # Otherwise, the UI may show empty/irrelevant execution trace.
    init.trace = []
    init.search_queries = []
    init.searches = []
    init.fetched_pages = {}
    init.critic = None
    init.critic_history = []
    init.plan = None
    init.plan_history = []
    init.reasoning_notes = []
    init.gaps = []
    init.proposed_queries = []
    init.execution_trace = init.execution_trace.__class__()

    try:
        raw_state = await graph.ainvoke(init, config={"recursion_limit": 120})
    except Exception as e:
        tb = traceback.format_exc(limit=50)
        logger.exception("/revise graph.ainvoke failed: %s", e)
        raise HTTPException(status_code=500, detail={"message": "revise failed", "error": str(e), "traceback": tb})

    final_state = ResearchState(**raw_state) if isinstance(raw_state, dict) else raw_state

    if final_state.public_report is None:
        final_state.public_report = PublicReport(id="rep_error", topic=req.query, content="", sources=prior_sources, word_count=0, created_at=None)

    # IMPORTANT: revision is a full new pass and MUST be re-judged.
    # Do not fall back to the previous/base evaluation.

    if final_state.judge_metadata is None:
        from app.core import settings

        final_state.judge_metadata = JudgeMetadata(
            evaluation_id=None,
            evaluated_at=None,
            judge_model=str(getattr(settings, "openai_model", "") or "").strip() or None,
            processing_time_ms=None,
            evaluation_version="1.0",
        )

    # Ensure the response carries the versioned id even if internal nodes rebuilt the report.
    if final_state.public_report is not None:
        final_state.public_report.id = next_id

    # Include a stable signal that the user's note was provided to the revision run.
    # (UI can show this in an "Improvements" banner.)
    meta = (final_state.judge_metadata.model_dump(mode="json") if final_state.judge_metadata is not None else {})
    meta["revision_user_note"] = (req.user_note or "").strip()
    meta["base_evaluation_id"] = str((req.metadata or {}).get("evaluation_id") or "").strip() or None

    # Build response and include execution details for the UI dashboard.
    # IMPORTANT: for /revise we must return the full report blocks (reporting-stage format)
    # under top-level `report`, and keep the judge public report under `public_report`.
    from app.graph.models import FinalReport

    final_report_obj = final_state.report
    if final_report_obj is None:
        # Best-effort fallback to empty blocks.
        final_report_obj = FinalReport(blocks=[], key_findings=[], evidence_and_sources=[], limitations=[])

    resp = JudgeResponse(
        status="complete",
        report=final_state.public_report,
        evaluation=final_state.judge_evaluation,
        metadata=meta,
    )

    # Attach trace + sources for dashboard parity with /research.
    try:
        resp = resp.model_copy(
            update={
                "execution_trace": getattr(final_state, "execution_trace", None).model_dump(mode="json")
                if getattr(final_state, "execution_trace", None) is not None
                else None
            }
        )
    except Exception:
        pass

    # Also attach raw trace lines + flattened sources for the existing right-side renderer.
    # (JudgeResponse is lenient; extra keys are OK for FastAPI JSON response.)
    payload = resp.model_dump(mode="json")

    # Provide the dashboard "report" object (blocks/limitations) expected by the frontend.
    payload["report"] = final_report_obj.model_dump(mode="json")

    # Also include the stable judge public report separately (useful for publish/ids).
    payload["public_report"] = final_state.public_report.model_dump(mode="json") if final_state.public_report is not None else None

    payload["trace"] = list(getattr(final_state, "trace", None) or [])
    payload["iteration_count"] = int(getattr(final_state, "iteration_count", 0) or 0)
    payload["critic"] = (
        getattr(final_state, "critic", None).model_dump(mode="json") if getattr(final_state, "critic", None) is not None else None
    )
    payload["plan"] = getattr(final_state, "plan", None).model_dump(mode="json") if getattr(final_state, "plan", None) is not None else None

    # Include both executed queries and flattened sources.
    payload["search_queries"] = list(getattr(final_state, "search_queries", None) or [])
    payload["sources"] = [
        r.model_dump(mode="json")
        for s in (getattr(final_state, "searches", None) or [])
        for r in (getattr(s, "results", None) or [])
    ]

    return payload


@app.post("/api/reports/{report_id}/publish")
async def publish_report(report_id: str, req: PublishRequest) -> dict:
    """Publish (persist) a report artifact to disk.

    Rules enforced server-side (unless DEBUG override is enabled):
    - evaluation.recommendation must be "publish"
    - factual_accuracy.score >= 7
    - completeness.score >= 7

    DEBUG override:
    - if env var DEBUG_ALLOW_PUBLISH=true, skip gating checks (still writes artifacts).

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

    def _truthy_env(name: str) -> bool:
        v = os.environ.get(name)
        if v is None:
            return False
        return str(v).strip().lower() in {"1", "true", "yes", "on"}

    debug_allow_publish = _truthy_env("DEBUG_ALLOW_PUBLISH")

    # Enforce publish gating (unless debug override is on).
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

    if (not debug_allow_publish) and (
        recommendation != "publish" or factual_i is None or complete_i is None or factual_i < 7 or complete_i < 7
    ):
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
                "debug": {
                    "DEBUG_ALLOW_PUBLISH": bool(debug_allow_publish),
                    "note": "Set DEBUG_ALLOW_PUBLISH=true in the API server environment to bypass gating.",
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
            yield f"data: {json.dumps({'type': 'state', 'stage': 'policy_guard', 'iteration_count': 0, 'trace': [], 'critic': None, 'status': 'processing', 'ts_ms': ts_ms()})}\n\n"

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
                if name not in {"policy_guard", "planning", "search", "reasoning", "critic_step", "report_step", "judge_step"}:
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
                            "status": getattr(state, "status", None),
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
                    if getattr(final_state_obj, "status", "") == "blocked":
                        response = {
                            "query": query,
                            "iteration_count": getattr(final_state_obj, "iteration_count", 0),
                            "trace": getattr(final_state_obj, "trace", None) or [],
                            "status": "blocked",
                            "final_user_message": getattr(final_state_obj, "final_user_message", "") or "",
                            "sources": [],
                        }
                        yield f"data: {json.dumps({'type': 'final', 'response': response})}\n\n"
                        return

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
