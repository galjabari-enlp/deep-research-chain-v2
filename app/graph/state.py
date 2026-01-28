from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    # Report
    report: Optional[FinalReport] = None

    # Trace
    trace: List[str] = field(default_factory=list)
