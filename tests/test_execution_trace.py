from __future__ import annotations

from app.graph.trace_models import ExecutionTrace
from app.main import ResearchRequest


def test_execution_trace_model_defaults() -> None:
    tr = ExecutionTrace()
    assert tr.iterations == []


def test_research_request_defaults() -> None:
    # sanity: request model still works (importing app.main)
    req = ResearchRequest(query="What is X?")
    assert req.max_revisions == 10
