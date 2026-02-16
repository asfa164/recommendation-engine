from __future__ import annotations

from pydantic import BaseModel, Field


class SimpleContext(BaseModel):
    persona: str | None = None
    domain: str | None = None
    instructions: str | None = None
    satisfactionCriteria: list[str] | None = None
    extraNotes: str | None = None


class SimpleObjectiveRequest(BaseModel):
    """
    Request payload.

    includeReason:
      - If true (default), the response includes a "reason" field.
      - If false, the response MUST NOT include a "reason" field.

    numRecommendations:
      - How many defining objectives to return (default 3, max 5).
    """
    objective: str = Field(..., min_length=1)
    context: SimpleContext | None = None

    includeReason: bool = True
    numRecommendations: int = Field(3, ge=1, le=5)


class SimpleRecommendResponse(BaseModel):
    """
    Response payload.

    reason is optional and will be omitted from the JSON response when includeReason=false
    (via FastAPI response_model_exclude_none=True).
    """
    reason: str | None = None
    definingObjectives: list[str]
