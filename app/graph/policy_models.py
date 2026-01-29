from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, conint, confloat, constr


PolicyCategory = Literal[
    "violence_poisoning",
    "violence_weapons_explosives",
    "violence_chemical_bio",
    "self_harm",
    "criminal_facilitation",
    "prompt_injection",
    "medical_dosing_restricted",
    "other",
]


class PolicyResult(BaseModel):
    blocked: bool = Field(
        ..., description="If true, the request must be refused and the graph must terminate early."
    )
    category: PolicyCategory = Field(..., description="Primary category driving the decision.")

    reason: constr(min_length=3, max_length=240) = Field(
        ..., description="Short non-sensitive reason for blocking/sanitization."
    )

    confidence: confloat(ge=0.0, le=1.0) = Field(
        default=0.75, description="Confidence in the safety decision."
    )

    sanitized_query: constr(min_length=0, max_length=700) = Field(
        default="",
        description="Sanitized version of the user query to pass downstream if allowed.",
    )

    safe_response: constr(min_length=0, max_length=1400) = Field(
        default="",
        description="If blocked, a safe refusal/helpful response to return to the user.",
    )

    tags: List[constr(min_length=1, max_length=40)] = Field(
        default_factory=list,
        description="Optional internal tags for debugging/analytics; must not leak policy text.",
    )


class PolicyState(BaseModel):
    status: Literal["blocked", "processing", "complete"] = "processing"
    policy: Optional[PolicyResult] = None

    # If blocked, we still want a stable user-facing payload.
    final_user_message: str = ""

    # Optional signal to tests/clients.
    blocked_code: Optional[conint(ge=100, le=599)] = None
