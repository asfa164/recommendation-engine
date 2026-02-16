from __future__ import annotations

import json
from typing import Any

from src.models.recommendation import SimpleObjectiveRequest, SimpleRecommendResponse
from src.inference.utilities import extract_text_from_anthropic_bedrock, safe_json_loads


SYSTEM_PROMPT_SIMPLE = """You are a helpful assistant that improves an objective into a clearer, testable defining objective.

Input: You will receive a JSON payload containing:
  - objective: string
  - context: optional object with fields like persona, domain, instructions, satisfactionCriteria, extraNotes

Output: You MUST return ONLY valid JSON with EXACTLY these keys:
{
  "reason": string,
  "suggestedDefiningObjective": string,
  "alternativeDefiningObjective": string
}

Do not wrap your JSON in markdown. Do not include any other keys.
"""


def recommend_objective(
    payload: dict | SimpleObjectiveRequest,
    bedrock_client: Any,
    model_id: str,
) -> SimpleRecommendResponse:
    """
    Main inference function used by the API route.

    - Validates payload using Pydantic.
    - Invokes Bedrock Anthropic-style request.
    - Extracts text from response, parses JSON, validates output model.
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
        "max_tokens": 512,
        "temperature": 0.0,
    }

    resp = bedrock_client.invoke_model(model_id=model_id, body=body)

    raw_text = extract_text_from_anthropic_bedrock(resp)
    if not raw_text:
        raise ValueError("Bedrock response did not contain model text")

    parsed = safe_json_loads(raw_text)
    return SimpleRecommendResponse.model_validate(parsed)
