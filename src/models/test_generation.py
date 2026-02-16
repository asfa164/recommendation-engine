from __future__ import annotations

from pydantic import BaseModel, Field


class TestGenContext(BaseModel):
    """
    Context used to generate test cases for a given domain.

    - description: free text describing product/bot/system behavior
    - language: output language of test cases (e.g. "en", "fr")
    - number_of_intents: how many intents / categories of tests to produce (default 3, max 10)
    - userDefinedVariables: any extra variables you want the model to consider (e.g. country, plan_type, channel)
    """
    description: str = Field(..., min_length=1)
    language: str = Field("en", min_length=2, max_length=16)
    number_of_intents: int = Field(3, ge=1, le=10)
    userDefinedVariables: dict = Field(default_factory=dict)


class TestGenerationRequest(BaseModel):
    domain: str = Field(..., min_length=1)
    context: TestGenContext


class GeneratedTestCase(BaseModel):
    """
    A generated test case in JSON form.
    Keep this flexible but structured enough for downstream use.
    """
    name: str
    description: str
    persona: str | None = None
    userVariables: dict = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)
    expected: list[str] = Field(default_factory=list)


class TestGenerationResponse(BaseModel):
    domain: str
    language: str
    testCases: list[GeneratedTestCase]
