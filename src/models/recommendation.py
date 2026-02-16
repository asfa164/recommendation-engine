from __future__ import annotations

from pydantic import BaseModel, Field


class SimpleContext(BaseModel):
    persona: str | None = None
    domain: str | None = None
    instructions: str | None = None
    satisfactionCriteria: list[str] | None = None
    extraNotes: str | None = None


class SimpleObjectiveRequest(BaseModel):
    objective: str = Field(..., min_length=1)
    context: SimpleContext | None = None


class SimpleRecommendResponse(BaseModel):
    reason: str
    suggestedDefiningObjective: str
    alternativeDefiningObjective: str
