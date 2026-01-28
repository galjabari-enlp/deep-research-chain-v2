from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, conint, constr

from app.graph.trace_models import ExecutionTrace


# --- Planning schema (MUST IMPLEMENT) ---


class SourceTypeRequirement(BaseModel):
    source_type: Literal[
        "official_government",
        "academic_peer_reviewed",
        "academic_preprint",
        "reputable_news",
        "industry_report",
        "company_blog",
        "encyclopedia",
        "other",
    ] = Field(..., description="Category of source desired for this research.")
    rationale: constr(min_length=10, max_length=400) = Field(
        ..., description="Why this source type is needed for this query."
    )
    must_have: bool = Field(
        True, description="If true, insufficient research if no sources of this type are found."
    )


class PlanStep(BaseModel):
    step_id: constr(pattern=r"^S[0-9]+$") = Field(..., description="Stable ID like S1, S2...")
    objective: constr(min_length=10, max_length=300)
    topics: List[constr(min_length=2, max_length=80)] = Field(
        ..., min_length=1, max_length=8,
        description="Topic keywords that constrain search. Used to prevent straying.",
    )
    must_find: List[constr(min_length=5, max_length=200)] = Field(
        default_factory=list,
        description="Concrete items to locate (e.g., '2023 revenue figure', 'regulatory citation').",
    )
    suggested_queries: List[constr(min_length=5, max_length=140)] = Field(
        ..., min_length=2, max_length=6,
        description="Search queries to run for this step. Should contain step topics.",
    )
    preferred_source_types: List[
        Literal[
            "official_government",
            "academic_peer_reviewed",
            "academic_preprint",
            "reputable_news",
            "industry_report",
            "company_blog",
            "encyclopedia",
            "other",
        ]
    ] = Field(default_factory=list)
    freshness_need: Literal["any", "last_30_days", "last_year", "last_5_years"] = "any"


class SuccessCriterion(BaseModel):
    criterion_id: constr(pattern=r"^C[0-9]+$") = Field(..., description="Stable ID like C1, C2...")
    description: constr(min_length=10, max_length=220)
    must_be_met: bool = True


class ResearchPlan(BaseModel):
    plan_version: conint(ge=1, le=50) = Field(..., description="Increment if replanned.")
    user_query: constr(min_length=3, max_length=500)

    research_objective: constr(min_length=10, max_length=400)
    scope_inclusions: List[constr(min_length=3, max_length=120)] = Field(
        default_factory=list, max_length=10
    )
    scope_exclusions: List[constr(min_length=3, max_length=120)] = Field(
        default_factory=list, max_length=10
    )

    key_terms: List[constr(min_length=2, max_length=40)] = Field(..., min_length=3, max_length=20)
    sub_questions: List[constr(min_length=10, max_length=200)] = Field(
        ..., min_length=2, max_length=8
    )

    source_type_requirements: List[SourceTypeRequirement] = Field(..., min_length=1, max_length=7)
    steps: List[PlanStep] = Field(..., min_length=2, max_length=6)

    success_criteria: List[SuccessCriterion] = Field(..., min_length=2, max_length=8)

    initial_search_batch: List[constr(min_length=5, max_length=140)] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="Deduplicated union of step suggested queries, capped.",
    )

    notes_for_search_agent: constr(min_length=0, max_length=600) = ""


# --- Critic schema (MUST IMPLEMENT) ---


class EvidenceQuality(BaseModel):
    url: constr(min_length=8, max_length=500)
    title: constr(min_length=1, max_length=200)
    quality: Literal["high", "medium", "low"]
    reasons: List[constr(min_length=5, max_length=180)] = Field(..., min_length=1, max_length=5)


class MissingGap(BaseModel):
    gap_id: constr(pattern=r"^G[0-9]+$")
    description: constr(min_length=10, max_length=220)
    mapped_to_success_criteria: List[constr(pattern=r"^C[0-9]+$")] = Field(default_factory=list)
    severity: Literal["blocking", "important", "minor"] = "important"
    suggested_query: constr(min_length=5, max_length=140)
    required_source_types: List[
        Literal[
            "official_government",
            "academic_peer_reviewed",
            "academic_preprint",
            "reputable_news",
            "industry_report",
            "company_blog",
            "encyclopedia",
            "other",
        ]
    ] = Field(default_factory=list)


class PlanIssue(BaseModel):
    issue: constr(min_length=10, max_length=220)
    why_it_matters: constr(min_length=10, max_length=220)
    fix: constr(min_length=10, max_length=220)


class CriticAssessment(BaseModel):
    iteration: conint(ge=1, le=10)

    decision: Literal["report", "refine_search", "replan"] = Field(
        ..., description="report if sufficient; refine_search if gaps remain; replan if plan flawed."
    )

    sufficiency_score: conint(ge=0, le=100)

    what_we_have: List[constr(min_length=10, max_length=240)] = Field(default_factory=list, max_length=10)

    evidence_quality: List[EvidenceQuality] = Field(default_factory=list, max_length=12)

    unmet_success_criteria: List[constr(pattern=r"^C[0-9]+$")] = Field(default_factory=list)

    missing_gaps: List[MissingGap] = Field(default_factory=list, max_length=8)

    plan_issues: List[PlanIssue] = Field(default_factory=list, max_length=6)

    feedback_to_planning: constr(min_length=0, max_length=600) = ""

    constraints_for_next_search: List[constr(min_length=3, max_length=120)] = Field(
        default_factory=list
    )


# --- Search result storage models ---


class SearchResult(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    url: HttpUrl
    snippet: str = Field(default="", max_length=2000)
    source: str = Field(default="serper", max_length=50)


class SearchQueryRecord(BaseModel):
    query: str = Field(..., min_length=3, max_length=200)
    executed_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    rationale: str = Field(default="", max_length=800)
    results: List[SearchResult] = Field(default_factory=list)


class ReportCitation(BaseModel):
    id: str = Field(..., min_length=1, max_length=80)
    url: HttpUrl
    title: str = Field(default="", max_length=300)


class ReportBlock(BaseModel):
    id: str = Field(..., min_length=1, max_length=80)
    heading: str = Field(default="", max_length=120)
    text: str = Field(..., min_length=1, max_length=4000)
    citations: List[ReportCitation] = Field(default_factory=list, max_length=12)


class FinalReport(BaseModel):
    # Backwards-compatible fields (legacy UI)
    key_findings: List[str] = Field(default_factory=list)
    evidence_and_sources: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

    # New UI contract: paragraph blocks with per-block citations
    blocks: List[ReportBlock] = Field(default_factory=list, max_length=40)


class ResearchResponse(BaseModel):
    query: str
    plan: Optional[ResearchPlan] = None
    iteration_count: int
    report: FinalReport
    sources: List[SearchResult] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)

    # Structured, sectioned execution trace (preferred by frontend)
    execution_trace: ExecutionTrace = Field(default_factory=lambda: ExecutionTrace())

    # UI helpers (optional)
    critic: Optional[CriticAssessment] = None
