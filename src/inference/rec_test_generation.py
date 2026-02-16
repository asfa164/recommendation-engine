from __future__ import annotations

import json
from typing import Any

from src.docs.descriptions import SYSTEM_PROMPT_JSON_REPAIR, SYSTEM_PROMPT_TEST_GENERATION
from src.inference.utilities import extract_text_from_anthropic_bedrock, safe_json_loads
from src.models.test_generation import TestGenerationRequest, TestGenerationResponse


def validate_min_counts(parsed: dict, min_cases: int) -> None:
    tcs = parsed.get("testCases")
    if not isinstance(tcs, list) or not tcs:
        raise ValueError("Model output must include non-empty 'testCases' list")
    if len(tcs) < min_cases:
        raise ValueError(f"Model returned {len(tcs)} testCases but minimum required is {min_cases}")


def invoke_bedrock_text(bedrock_client: Any, model_id: str, system: str, user_text: str, max_tokens: int) -> str:
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


def parse_or_repair_json(raw_text: str, bedrock_client: Any, model_id: str) -> dict:
    try:
        return safe_json_loads(raw_text)
    except Exception:
        repaired_text = invoke_bedrock_text(
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

    gen_text = invoke_bedrock_text(
        bedrock_client=bedrock_client,
        model_id=model_id,
        system=SYSTEM_PROMPT_TEST_GENERATION,
        user_text=json.dumps(model_input, ensure_ascii=False, indent=2),
        max_tokens=1100,
    )
    if not gen_text:
        raise ValueError("Bedrock response did not contain model text")

    parsed = parse_or_repair_json(gen_text, bedrock_client=bedrock_client, model_id=model_id)
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")

    validate_min_counts(parsed, min_cases=req.context.number_of_intents)

    # Force request truth (prevents model drift)
    parsed["domain"] = req.domain
    parsed["language"] = req.context.language

    return TestGenerationResponse.model_validate(parsed)
