from __future__ import annotations

import json
from typing import Any

from src.inference.utilities import extract_text_from_anthropic_bedrock, safe_json_loads
from src.models.test_generation import TestGenerationRequest, TestGenerationResponse


SYSTEM_PROMPT_TEST_GENERATION = """You are a helpful assistant that generates high-quality chatbot test cases in JSON.

Input:
You will receive a JSON payload containing:
- domain: string
- context:
  - description: string
  - language: string (e.g. "en")
  - number_of_intents: integer (1..10)
  - userDefinedVariables: object (arbitrary key/value variables)

Output rules:
- You MUST return ONLY valid JSON. No markdown. No extra keys.
- The JSON MUST match EXACTLY this schema:

{
  "domain": string,
  "language": string,
  "testCases": [
    {
      "name": string,
      "description": string,
      "persona": string | null,
      "userVariables": object,
      "steps": [string, ...],
      "expected": [string, ...]
    }
  ]
}

Quality requirements:
- Return AT LEAST (number_of_intents) test cases.
- Each test case should represent a different intent/category relevant to the domain.
- Steps should be concrete and testable, written in the requested language.
- Expected should include what a good bot/system should do (clarifying questions, constraints, confirmations, etc.).
"""


def _validate_min_counts(parsed: dict, min_cases: int) -> None:
    tcs = parsed.get("testCases")
    if not isinstance(tcs, list) or not tcs:
        raise ValueError("Model output must include non-empty 'testCases' list")
    if len(tcs) < min_cases:
        raise ValueError(f"Model returned {len(tcs)} testCases but minimum required is {min_cases}")


def generate_test_cases(
    payload: dict | TestGenerationRequest,
    bedrock_client: Any,
    model_id: str,
) -> TestGenerationResponse:
    req = payload if isinstance(payload, TestGenerationRequest) else TestGenerationRequest.model_validate(payload)

    model_input = req.model_dump()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT_TEST_GENERATION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(model_input, ensure_ascii=False, indent=2)}
                ],
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.2,
    }

    resp = bedrock_client.invoke_model(model_id=model_id, body=body)
    raw_text = extract_text_from_anthropic_bedrock(resp)
    if not raw_text or not raw_text.strip():
        raise ValueError("Bedrock response did not contain model text")

    parsed = safe_json_loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")

    # Enforce minimum number of test cases based on requested number_of_intents
    _validate_min_counts(parsed, min_cases=req.context.number_of_intents)

    # Ensure domain/language present (and prefer request if model deviates)
    parsed["domain"] = req.domain
    parsed["language"] = req.context.language

    return TestGenerationResponse.model_validate(parsed)
