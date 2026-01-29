from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conint, confloat, constr

# NOTE: These models represent the *public* evaluation payload required by the
# "Judge" step. They are intentionally separate from internal graph state models.


class PublicReport(BaseModel):
    id: constr(min_length=1, max_length=80)
    topic: constr(min_length=1, max_length=500)
    content: constr(min_length=0, max_length=200_000) = ""
    sources: List[dict] = Field(default_factory=list)
    word_count: conint(ge=0) = 0
    created_at: Optional[str] = None


class JudgeCategory(BaseModel):
    score: conint(ge=0, le=10)
    max_score: conint(ge=1, le=10) = 10
    percentage: conint(ge=0, le=100)
    reasoning: constr(min_length=1, max_length=1600)
    strengths: List[constr(min_length=2, max_length=240)] = Field(default_factory=list, max_length=12)
    weaknesses: List[constr(min_length=2, max_length=240)] = Field(default_factory=list, max_length=12)


CoverageStatus = Literal["covered", "missing", "not_applicable"]


class CoverageItem(BaseModel):
    status: CoverageStatus = "missing"
    notes: constr(min_length=0, max_length=240) = ""

    @property
    def covered(self) -> bool:  # back-compat convenience for internal logic
        return self.status == "covered"


class CompletenessCategory(JudgeCategory):
    coverage: Dict[str, CoverageItem] = Field(default_factory=dict)


Recommendation = Literal["publish", "revise", "reject"]


class JudgeEvaluation(BaseModel):
    factual_accuracy: JudgeCategory
    completeness: CompletenessCategory

    overall_score: confloat(ge=0, le=10)
    grade: constr(min_length=1, max_length=4)
    overall_assessment: constr(min_length=10, max_length=2000)

    recommendation: Recommendation
    confidence: confloat(ge=0, le=1)

    flags: List[constr(min_length=3, max_length=80)] = Field(default_factory=list, max_length=20)
    suggested_improvements: List[constr(min_length=5, max_length=220)] = Field(default_factory=list, max_length=20)


class JudgeMetadata(BaseModel):
    evaluation_id: Optional[str] = None
    evaluated_at: Optional[str] = None
    judge_model: constr(min_length=2, max_length=120) = "claude-sonnet-4-20250514"
    processing_time_ms: Optional[conint(ge=0)] = None
    evaluation_version: constr(min_length=1, max_length=20) = "1.0"


class JudgeResponse(BaseModel):
    status: Literal["processing", "complete", "blocked"]
    report: PublicReport
    evaluation: Optional[JudgeEvaluation] = None
    metadata: JudgeMetadata
    # Optional: include structured execution trace for UI dashboards (useful for /revise).
    execution_trace: Optional[dict] = None


# ---- LLM-only schema ----
# We ask the LLM to fill only the "human" fields and raw scores.
# The backend deterministically computes:
# - percentage fields
# - overall_score
# - grade
# - recommendation
# - evaluation_id/evaluated_at/processing_time_ms


class JudgeLLMCategory(BaseModel):
    # NOTE: We intentionally allow wider ranges here so we can parse even slightly
    # invalid LLM outputs and then clamp deterministically in backend post-processing.
    score: conint(ge=-100, le=100)
    reasoning: constr(min_length=1, max_length=1600)
    strengths: List[constr(min_length=2, max_length=240)] = Field(default_factory=list, max_length=12)
    weaknesses: List[constr(min_length=2, max_length=240)] = Field(default_factory=list, max_length=12)


class JudgeLLMCompleteness(JudgeLLMCategory):
    coverage: Dict[str, CoverageItem] = Field(default_factory=dict)


class JudgeLLMOutput(BaseModel):
    factual_accuracy: JudgeLLMCategory
    completeness: JudgeLLMCompleteness

    overall_assessment: constr(min_length=10, max_length=2000)
    flags: List[constr(min_length=3, max_length=80)] = Field(default_factory=list, max_length=20)
    suggested_improvements: List[constr(min_length=5, max_length=220)] = Field(default_factory=list, max_length=20)

    # LLM-provided confidence can be used as a hint; backend will clamp and may adjust.
    confidence: Optional[confloat(ge=0, le=1)] = None


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
