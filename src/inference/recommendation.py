from __future__ import annotations

import json
from typing import Any

from src.models.recommendation import SimpleObjectiveRequest, SimpleRecommendResponse
from src.inference.utilities import extract_text_from_anthropic_bedrock, safe_json_loads


SYSTEM_PROMPT_SIMPLE = """You are a helpful assistant that improves an objective into clearer, testable defining objectives.

Input: You will receive a JSON payload containing:
  - objective: string
  - context: optional object with fields like persona, domain, instructions, satisfactionCriteria, extraNotes
  - includeReason: boolean (default true)
  - numRecommendations: integer (1..5, default 3)

Output rules:
- You MUST return ONLY valid JSON (no markdown, no extra text).
- You MUST return EXACTLY these keys depending on includeReason:

If includeReason is true:
{
  "reason": string,
  "definingObjectives": [string, ...]   // length MUST equal numRecommendations
}

If includeReason is false:
{
  "definingObjectives": [string, ...]   // length MUST equal numRecommendations
}

No other keys are allowed. The definingObjectives list MUST contain EXACTLY numRecommendations items.
"""


def _validate_and_shape_output(
    parsed: dict,
    include_reason: bool,
    num_recommendations: int,
) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")

    # Extract objectives
    defining = parsed.get("definingObjectives")
    if not isinstance(defining, list) or not all(isinstance(x, str) and x.strip() for x in defining):
        raise ValueError("Model output 'definingObjectives' must be a non-empty list of strings")

    # Enforce count (truncate if too many; fail if too few)
    if len(defining) < num_recommendations:
        raise ValueError(
            f"Model returned {len(defining)} definingObjectives but numRecommendations={num_recommendations}"
        )
    defining = [x.strip() for x in defining[:num_recommendations]]

    out: dict = {"definingObjectives": defining}

    if include_reason:
        reason = parsed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Model output missing required non-empty 'reason'")
        out["reason"] = reason.strip()

    # If includeReason is false, ensure reason is not included (even if model returned it)
    return out


def recommend_objective(
    payload: dict | SimpleObjectiveRequest,
    bedrock_client: Any,
    model_id: str,
) -> SimpleRecommendResponse:
    """
    Main inference function used by the API route.

    - Validates payload using Pydantic.
    - Instructs the model to return N defining objectives and optional reason.
    - Extracts model text, parses JSON, validates & shapes output.
    """
    req = payload if isinstance(payload, SimpleObjectiveRequest) else SimpleObjectiveRequest.model_validate(payload)
    model_input = req.model_dump()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT_SIMPLE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(model_input, ensure_ascii=False, indent=2),
                    }
                ],
            }
        ],
        "max_tokens": 768,
        "temperature": 0.0,
    }

    resp = bedrock_client.invoke_model(model_id=model_id, body=body)

    raw_text = extract_text_from_anthropic_bedrock(resp)
    if not raw_text:
        raise ValueError("Bedrock response did not contain model text")

    parsed = safe_json_loads(raw_text)

    shaped = _validate_and_shape_output(
        parsed=parsed,
        include_reason=req.includeReason,
        num_recommendations=req.numRecommendations,
    )

    return SimpleRecommendResponse.model_validate(shaped)
