from __future__ import annotations

import json


def extract_text_from_anthropic_bedrock(resp: dict) -> str:
    """
    Extract concatenated text from a Bedrock Anthropic-style response.

    Expected shape (Anthropic on Bedrock):
      { "content": [ { "type": "text", "text": "..." }, ... ] }

    Falls back to other common wrapper keys if needed.
    """
    content = resp.get("content")
    if isinstance(content, list):
        chunks: list[str] = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                chunks.append(c["text"])
        if chunks:
            return "".join(chunks).strip()

    # Some wrappers put the model text in different keys
    for key in ("outputText", "completion", "generation", "text"):
        val = resp.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""


def safe_json_loads(text: str) -> dict:
    """
    Parse JSON, with a small recovery attempt if the model included extra text.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Recovery: extract the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
