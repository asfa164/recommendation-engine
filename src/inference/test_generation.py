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

Output rules (STRICT):
- You MUST return ONLY valid JSON. No markdown. No extra text.
- All strings MUST be valid JSON strings (escape quotes like \\" and newlines like \\n).
- Do NOT include trailing commas.
- Do NOT include any keys other than: domain, language, testCases.

The JSON MUST match EXACTLY this schema:

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
- Steps and expected should be concrete and testable, written in the requested language.
"""


SYSTEM_PROMPT_JSON_REPAIR = """You are a strict JSON repair tool.

You will be given text that is intended to be JSON but may be invalid.
Your job is to output ONLY valid JSON that matches the required schema below.
No markdown, no commentary, no extra keys.

Required schema:

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

Rules:
- Output must be valid JSON.
- Escape any quotes inside strings properly.
- Remove trailing commas.
- Preserve as much original meaning/content as possible.
"""


def _validate_min_counts(parsed: dict, min_cases: int) -> None:
    tcs = parsed.get("testCases")
    if not isinstance(tcs, list) or not tcs:
        raise ValueError("Model output must include non-empty 'testCases' list")
    if len(tcs) < min_cases:
        raise ValueError(f"Model returned {len(tcs)} testCases but minimum required is {min_cases}")


def _invoke_bedrock_text(bedrock_client: Any, model_id: str, system: str, user_text: str, max_tokens: int) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = bedrock_client.invoke_model(model_id=model_id, body=body)
    raw = extract_text_from_anthropic_bedrock(resp)
    return (raw or "").strip()


def _parse_or_repair_json(
    raw_text: str,
    bedrock_client: Any,
    model_id: str,
) -> dict:
    """
    Try to parse JSON. If it fails, do a single repair pass via the model, then parse again.
    """
    try:
        return safe_json_loads(raw_text)
    except Exception:
        repaired_text = _invoke_bedrock_text(
            bedrock_client=bedrock_client,
            model_id=model_id,
            system=SYSTEM_PROMPT_JSON_REPAIR,
            user_text=raw_text,
            max_tokens=1400,
        )
        if not repaired_text:
            raise ValueError("Model returned invalid JSON and repair step returned empty text.")
        return safe_json_loads(repaired_text)


def generate_test_cases(
    payload: dict | TestGenerationRequest,
    bedrock_client: Any,
    model_id: str,
) -> TestGenerationResponse:
    req = payload if isinstance(payload, TestGenerationRequest) else TestGenerationRequest.model_validate(payload)
    model_input = req.model_dump()

    # First call: generate JSON
    gen_text = _invoke_bedrock_text(
        bedrock_client=bedrock_client,
        model_id=model_id,
        system=SYSTEM_PROMPT_TEST_GENERATION,
        user_text=json.dumps(model_input, ensure_ascii=False, indent=2),
        max_tokens=1100,
    )
    if not gen_text:
        raise ValueError("Bedrock response did not contain model text")

    parsed = _parse_or_repair_json(gen_text, bedrock_client=bedrock_client, model_id=model_id)
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")

    _validate_min_counts(parsed, min_cases=req.context.number_of_intents)

    # Force request truth (prevents model drift)
    parsed["domain"] = req.domain
    parsed["language"] = req.context.language

    return TestGenerationResponse.model_validate(parsed)
