from __future__ import annotations

from app.graph.models import ResearchPlan
from app.graph.state import ResearchState


def test_state_dataclass_defaults() -> None:
    st = ResearchState(query="x y z")
    assert st.query == "x y z"
    assert st.plan is None
    assert st.plan_history == []
    assert st.search_queries == []
    assert st.searches == []
    assert st.iteration_count == 0
    assert st.max_revisions == 10


def test_plan_schema_available() -> None:
    schema = ResearchPlan.model_json_schema()
    assert isinstance(schema, dict)
    assert schema.get("title") == "ResearchPlan"
