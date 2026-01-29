from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.graph.judge_models import JudgeEvaluation, JudgeMetadata, PublicReport
from app.graph.trace_models import ExecutionTrace

from .models import (
    CriticAssessment,
    FinalReport,
    ResearchPlan,
    SearchQueryRecord,
)


@dataclass
class ResearchState:
    # Inputs
    query: str

    # Planning
    plan: Optional[ResearchPlan] = None
    plan_history: List[ResearchPlan] = field(default_factory=list)

    # Structured execution trace (iteration -> sections -> items)
    execution_trace: "ExecutionTrace" = field(default_factory=lambda: ExecutionTrace())

    # Map each critic iteration -> plan_version used for the subsequent search.
    # This helps test/diagnose replanning behavior.
    iteration_plan_versions: Dict[int, int] = field(default_factory=dict)

    # Search
    search_queries: List[str] = field(default_factory=list)
    searches: List[SearchQueryRecord] = field(default_factory=list)

    # Optional: fetched page text (if enabled)
    fetched_pages: Dict[str, str] = field(default_factory=dict)  # url -> extracted text

    # Reasoning
    reasoning_notes: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    proposed_queries: List[str] = field(default_factory=list)

    # Critic
    critic: Optional[CriticAssessment] = None
    critic_history: List[CriticAssessment] = field(default_factory=list)

    # Loop control
    iteration_count: int = 0
    max_revisions: int = 10

    # Report (legacy)
    report: Optional[FinalReport] = None

    # Report (public, stable contract for downstream judge + future publish/revision UX)
    public_report: Optional[PublicReport] = None

    # Judge output (latest) + metadata
    judge_evaluation: Optional[JudgeEvaluation] = None
    judge_metadata: Optional[JudgeMetadata] = None

    # Revision-loop support (future): persist past evaluations + user feedback
    prior_evaluations: List[JudgeEvaluation] = field(default_factory=list)
    user_revision_requests: List[str] = field(default_factory=list)

    # Trace
    trace: List[str] = field(default_factory=list)
