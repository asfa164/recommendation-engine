from __future__ import annotations

import json
import re
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
- The first non-whitespace character MUST be '{' and the last MUST be '}'.
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


def parse_objectives_from_text(text: str, n: int) -> list[str]:
    """
    Best-effort fallback parser for non-JSON model outputs.
    Extracts numbered/bulleted lines as objectives.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # patterns: "1. ...", "1) ...", "- ...", "• ...", "* ..."
    obj: list[str] = []
    for ln in lines:
        m = re.match(r"^(\d+[\.\)]\s+|[-•*]\s+)(.+)$", ln)
        if m:
            candidate = m.group(2).strip()
            if candidate:
                obj.append(candidate)

    # If no bullet/numbered list, try splitting by semicolons as last resort
    if not obj:
        parts = [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]
        # take only reasonably long parts
        obj = [p for p in parts if len(p) >= 12]

    # de-dup, preserve order
    seen = set()
    deduped: list[str] = []
    for x in obj:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(x)

    return deduped[:n]


def fallback_parse_non_json_output(raw_text: str, include_reason: bool, n: int) -> dict:
    """
    Convert a plain-text model output into the expected dict shape.
    """
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Model returned empty text (expected JSON).")

    objectives = parse_objectives_from_text(cleaned, n)
    if len(objectives) < n:
        raise ValueError(
            f"Model output was not valid JSON and fallback parsing found only {len(objectives)} "
            f"objective(s) but numRecommendations={n}. Raw start: {cleaned[:120]!r}"
        )

    out: dict = {"definingObjectives": objectives[:n]}

    if include_reason:
        # Use the first non-empty paragraph as reason (before a list if possible)
        # Split on a blank line; take first chunk
        chunks = re.split(r"\n\s*\n", cleaned)
        reason_candidate = chunks[0].strip()

        # If first chunk looks like a list item, make a generic reason instead
        if re.match(r"^(\d+[\.\)]\s+|[-•*]\s+)", reason_candidate):
            reason_candidate = "Generated defining objectives based on the provided objective and context."

        out["reason"] = reason_candidate[:600].strip()

    return out


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

    # If includeReason is false, ensure reason is not included (even if model returned it)
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

    if not raw_text or not raw_text.strip():
        raise ValueError("Bedrock response did not contain model text")

    # 1) Try JSON parse
    try:
        parsed = safe_json_loads(raw_text)
        shaped = validate_and_shape_output(
            parsed=parsed,
            include_reason=req.includeReason,
            num_recommendations=req.numRecommendations,
        )
    except Exception:
        # 2) Fallback parse for non-JSON model output
        shaped = fallback_parse_non_json_output(
            raw_text=raw_text,
            include_reason=req.includeReason,
            n=req.numRecommendations,
        )

    return SimpleRecommendResponse.model_validate(shaped)
