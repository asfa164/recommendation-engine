from __future__ import annotations

import json
from typing import Any

from src.docs.descriptions import SYSTEM_PROMPT_OBJECTIVE
from src.inference.utilities import extract_text_from_anthropic_bedrock, safe_json_loads
from src.models.recommendation import SimpleObjectiveRequest, SimpleRecommendResponse


def validate_and_shape_output(parsed: dict, include_reason: bool, num_recommendations: int) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")

    defining = parsed.get("definingObjectives")
    if not isinstance(defining, list) or not all(isinstance(x, str) and x.strip() for x in defining):
        raise ValueError("Model output 'definingObjectives' must be a non-empty list of strings")

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

    return out


def recommend_objective(
    payload: dict | SimpleObjectiveRequest,
    bedrock_client: Any,
    model_id: str,
) -> SimpleRecommendResponse:
    req = payload if isinstance(payload, SimpleObjectiveRequest) else SimpleObjectiveRequest.model_validate(payload)
    model_input = req.model_dump()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT_OBJECTIVE,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(model_input, ensure_ascii=False, indent=2)}],
            }
        ],
        "max_tokens": 768,
        "temperature": 0.0,
    }

    resp = bedrock_client.invoke_model(model_id=model_id, body=body)
    raw_text = extract_text_from_anthropic_bedrock(resp)
    if not raw_text or not raw_text.strip():
        raise ValueError("Bedrock response did not contain model text")

    parsed = safe_json_loads(raw_text)
    shaped = validate_and_shape_output(
        parsed=parsed,
        include_reason=req.includeReason,
        num_recommendations=req.numRecommendations,
    )

    return SimpleRecommendResponse.model_validate(shaped)
