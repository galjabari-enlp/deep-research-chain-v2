# Deep Research Agent (LangGraph + FastAPI + Reflection Loop)

A runnable backend repo implementing a **cyclical LangGraph research agent** with a **reflection/critic** self-correction loop.

Key properties:
- **FastAPI** backend with `POST /research` and `GET /health`
- **LangGraph** cyclical workflow:
  [`planning_node()`](app/graph/nodes/planning.py:1) → [`search_node()`](app/graph/nodes/search.py:1) → [`reasoning_node()`](app/graph/nodes/reasoning.py:1) → [`critic_node()`](app/graph/nodes/critic.py:1) → (conditional: replan/refine/report) → [`report_node()`](app/graph/nodes/report.py:1)
- **Serper** search (snippets only) via `https://google.serper.dev/search`
- **OpenAI-compatible** LLM via `OPENAI_BASE_URL` + `OPENAI_API_KEY`
- Strict **structured outputs** for planning + critic (Pydantic-validated, with one re-ask)

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
  "query": "Explain the main causes of inflation in 2021-2023",
  "plan": { "plan_version": 1, "research_objective": "…" },
  "iteration_count": 3,
  "report": {
    "key_findings": ["…"],
    "evidence_and_sources": ["…"],
    "limitations": ["This agent uses search snippets only …"]
  },
  "sources": [{"title":"…","url":"…","snippet":"…"}],
  "trace": ["Planned v1 …", "Searched: …", "Critic decision: …"]
}
```

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
