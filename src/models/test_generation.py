from __future__ import annotations

from pydantic import BaseModel, Field


class TestGenContext(BaseModel):
    description: str = Field(
        ...,
        min_length=1,
        description="Description of the bot/system and what kinds of test cases to generate.",
    )
    language: str = Field(
        "en",
        description="Language of generated test cases (e.g. 'en').",
        min_length=2,
        max_length=16,
    )
    number_of_intents: int = Field(
        3,
        ge=1,
        le=10,
        description="Minimum number of intent categories / test cases to generate (default 3, max 10).",
    )
    userDefinedVariables: dict = Field(
        default_factory=dict,
        description="Extra variables that should be applied when generating tests (e.g. country, segment, channel).",
    )


class TestGenerationRequest(BaseModel):
    domain: str = Field(..., min_length=1, description="Domain identifier (e.g. telecom_billing).")
    context: TestGenContext = Field(..., description="Context object (description is required; other fields optional).")


class GeneratedTestCase(BaseModel):
    name: str = Field(..., description="Human-readable name of the test case.")
    description: str = Field(..., description="Short description of what the test covers.")
    persona: str | None = Field(default=None, description="Optional persona for the test.")
    userVariables: dict = Field(default_factory=dict, description="Variables used in the test case.")
    steps: list[str] = Field(default_factory=list, description="Steps for the test conversation.")
    expected: list[str] = Field(default_factory=list, description="Expected bot behaviors/outcomes.")


class TestGenerationResponse(BaseModel):
    domain: str = Field(..., description="Domain identifier.")
    language: str = Field(..., description="Language of generated test cases.")
    testCases: list[GeneratedTestCase] = Field(..., description="List of generated test cases.")
