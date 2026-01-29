# Deep Research Agent (LangGraph + FastAPI + Reflection Loop)

A runnable backend repo implementing a **cyclical LangGraph research agent** with a **reflection/critic** self-correction loop.

Key properties:
- **FastAPI** backend with `POST /research` and `GET /health`
- **LangGraph** cyclical workflow:
  [`planning_node()`](app/graph/nodes/planning.py:1) → [`search_node()`](app/graph/nodes/search.py:1) → [`reasoning_node()`](app/graph/nodes/reasoning.py:1) → [`critic_node()`](app/graph/nodes/critic.py:1) → (conditional: replan/refine/report) → [`report_node()`](app/graph/nodes/report.py:1) → [`judge_node()`](app/graph/nodes/judge.py:1)
- **Serper** search (snippets only) via `https://google.serper.dev/search`
- **OpenAI-compatible** LLM via `OPENAI_BASE_URL` + `OPENAI_API_KEY`
- Strict **structured outputs** for planning + critic + judge (Pydantic-validated, with one re-ask)

## Architecture overview

### State
The graph state is stored in [`ResearchState`](app/graph/state.py:1) and includes:
- `query`
- `plan`, `plan_history`
- `search_queries`, `searches` (snippets + URLs)
- `reasoning_notes`, `gaps`, `proposed_queries`
- `critic`, `critic_history`
- `iteration_count`, `max_revisions`
- `report`, `trace`

### Reflection loop
- The critic returns a decision JSON (`report` / `refine_search` / `replan`).
- Conditional routing is implemented in [`_route_from_critic()`](app/graph/build_graph.py:12).
- A hard stop occurs after `max_revisions` iterations; the agent produces a best-effort report.

### Safety: “Search must not stray from the plan”
Search queries are validated against plan step topics:
- Queries must contain at least one topic keyword (substring match).
- Critic constraints (if present) further restrict queries.
- Rejected queries are recorded in `state.trace`.

Implementation lives in:
- Plan topic extraction / matching: [`plan_all_topics()`](app/graph/enforce.py:33)
- Query validation: [`_validate_query_against_plan()`](app/graph/nodes/search.py:27)

## Setup

### 1) Create a virtualenv and install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -e ".[test]"
```

### 2) Configure environment

Copy `.env.example` to `.env` and fill in keys:

- `OPENAI_API_KEY` – API key
- `OPENAI_BASE_URL` – base URL for an OpenAI-compatible API (leave empty for OpenAI default)
- `OPENAI_MODEL` – model name (default `gpt-4o-mini`)
- `SERPER_API_KEY` – Serper key

See: [`.env.example`](.env.example)

## Run

### API

```bash
uvicorn app.main:app --reload
```

Endpoints:
- `GET /health` → `{ "status": "ok" }`
- `POST /research`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain the main causes of inflation in 2021-2023","max_revisions":5}'
```

Example response (truncated):

```json
{
  "status": "complete",
  "report": {
    "id": "rep_...",
    "topic": "Explain the main causes of inflation in 2021-2023",
    "content": "...",
    "sources": [],
    "word_count": 0,
    "created_at": null
  },
  "evaluation": {
    "factual_accuracy": {"score": 8, "max_score": 10, "percentage": 80, "reasoning": "...", "strengths": [], "weaknesses": []},
    "completeness": {"score": 7, "max_score": 10, "percentage": 70, "reasoning": "...", "strengths": [], "weaknesses": [], "coverage": {"overview": {"covered": true, "notes": ""}}},
    "overall_score": 7.65,
    "grade": "C",
    "overall_assessment": "...",
    "recommendation": "publish",
    "confidence": 0.7,
    "flags": [],
    "suggested_improvements": []
  },
  "metadata": {
    "evaluation_id": "...",
    "evaluated_at": "2026-01-01T00:00:00Z",
    "judge_model": "claude-sonnet-4-20250514",
    "processing_time_ms": 1234,
    "evaluation_version": "1.0"
  }
}
```

### Judge scoring + gating logic

Implemented in [`build_judge_response()`](app/graph/nodes/judge.py:171).

- Hard thresholds (publication "greenlight"):
  - If `factual_accuracy.score < 7` → **NOT publishable** → recommendation is `revise` (or `reject` for severe flags / very low accuracy)
  - If `completeness.score < 7` → **NOT publishable** → recommendation is `revise`

- Deterministic overall score formula (accuracy-weighted):
  - `overall_score = 0.65 * factual_accuracy.score + 0.35 * completeness.score`

- Grade mapping:
  - 9–10: `A`
  - 8–8.99: `B`
  - 7–7.99: `C`
  - 6–6.99: `D`
  - <6: `F`

- `evaluation.completeness.coverage` is **template-based** (stable keys) so the frontend can render it consistently.

### CLI

```bash
python -m app.cli "Explain the main causes of inflation in 2021-2023" --max-revisions 5
```

The CLI prints:
- iteration trace
- plan objective + initial batch
- queries executed
- final critic JSON
- final report

## Tests

```bash
pytest
```

## Notes / limitations
- **Snippets-only**: the agent does not scrape full pages.
- Evidence is limited by what Serper returns in snippets.
- LLM structured-output failures are handled via one re-ask; if still invalid, a best-effort report is produced.
