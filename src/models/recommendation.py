from __future__ import annotations

from pydantic import BaseModel, Field


class SimpleContext(BaseModel):
    persona: str | None = Field(default=None, description="User persona (optional).")
    domain: str | None = Field(default=None, description="Domain identifier (optional).")
    instructions: str | None = Field(default=None, description="Extra instructions for the model (optional).")
    satisfactionCriteria: list[str] | None = Field(
        default=None,
        description="List of criteria the output should satisfy (optional).",
    )
    extraNotes: str | None = Field(default=None, description="Any extra notes (optional).")


class SimpleObjectiveRequest(BaseModel):
    """
    Request payload for /{env}/recommendation.
    """
    objective: str = Field(..., min_length=1, description="Vague objective to refine into clearer defining objective(s).")
    context: SimpleContext | None = Field(default=None, description="Optional context object (all fields are optional).")

    includeReason: bool = Field(
        True,
        description="If true, include 'reason' in the response. If false, 'reason' is omitted.",
    )
    numRecommendations: int = Field(
        3,
        ge=1,
        le=5,
        description="Number of defining objectives to return (default 3, max 5).",
    )


class SimpleRecommendResponse(BaseModel):
    reason: str | None = Field(
        default=None,
        description="Explanation of why the defining objective(s) were suggested (omitted when includeReason=false).",
    )
    definingObjectives: list[str] = Field(..., description="List of defining objectives (length = numRecommendations).")
