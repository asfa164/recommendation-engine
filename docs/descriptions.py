from __future__ import annotations

# =========================
# OpenAPI / Swagger text
# =========================

API_DESCRIPTION = """
Single-purpose API providing two endpoints:

### 1) Recommendation
**Endpoint:** `/{env}/recommendation`

- **Required:** `objective` (string)
- **Optional:** `context` (object; all fields inside are optional)
- **Optional:** `includeReason` (boolean, default `true`)
- **Optional:** `numRecommendations` (int, default `3`, max `5`)

### 2) Test Generation
**Endpoint:** `/{env}/test-generation`

- **Required:** `domain` (string)
- **Required:** `context` (object)
- **Required (inside context):** `description` (string)
- **Optional (inside context):** `language` (string, default `en`)
- **Optional (inside context):** `number_of_intents` (int, default `3`, max `10`)
- **Optional (inside context):** `userDefinedVariables` (object)

**Authentication:** requires `X-API-Key` header.
"""

RECOMMENDATION_DESC = """
Lightweight endpoint for defining objective recommendations.

**Endpoint:** `/{env}/recommendation`

**Required field:**
- `objective` (string)

**Optional fields:**
- `context` (object). You may omit it entirely.
- `includeReason` (boolean, default `true`)
- `numRecommendations` (int, default `3`, max `5`)

**All fields inside** `context` **are optional:**
- `persona`
- `domain`
- `instructions`
- `satisfactionCriteria` (list of strings)
- `extraNotes`
- `userVariables` (object; optional user-defined variables)

**Authentication:** requires `X-API-Key` header.
"""

TEST_GEN_DESC = """
Generate structured test cases for a given domain.

**Endpoint:** `/{env}/test-generation`

**Required fields:**
- `domain` (string)
- `context` (object)

**Required field inside** `context`:
- `description` (string)

**Optional fields inside** `context`:
- `language` (string, default `en`)
- `number_of_intents` (int, default `3`, max `10`)
- `userDefinedVariables` (object)

**Authentication:** requires `X-API-Key` header.
"""


# =========================
# Swagger request examples
# =========================

RECOMMENDATION_EXAMPLES = {
    "minimal": {
        "summary": "Minimal (objective only)",
        "description": "Smallest valid request body.",
        "value": {"objective": "What is this extra charge?"},
    },
    "with_context": {
        "summary": "With context",
        "description": "Adds context to guide the model response.",
        "value": {
            "objective": "What is this extra charge?",
            "context": {
                "persona": "Postpaid telecom customer in Ireland",
                "domain": "telecom_billing",
                "instructions": "Treat this as a vague billing query. Focus on eliciting details before promising a resolution.",
                "satisfactionCriteria": [
                    "Acknowledge the concern about the extra charge.",
                    "Ask for a specific detail (date/amount/invoice ID).",
                    "Avoid confirming the cause before checking bill details.",
                ],
                "extraNotes": "Customer is confused but not angry. Keep tone calm and reassuring.",
            },
        },
    },
    "with_user_variables": {
        "summary": "With userVariables",
        "description": "Include optional userVariables to guide the recommendation.",
        "value": {
            "objective": "Act as customer trying to return a non-returnable item",
            "context": {
                "persona": "Angry customer",
                "domain": "ecommerce",
                "instructions": "Customer is a first-time user not aware of return policy.",
                "userVariables": {"order_number": "1234"},
            },
        },
    },
    "no_reason_more_results": {
        "summary": "No reason + 5 recommendations",
        "description": "Omit reason from response and return 5 defining objectives.",
        "value": {
            "objective": "Help me dispute this roaming charge.",
            "includeReason": False,
            "numRecommendations": 5,
            "context": {"domain": "telecom_billing"},
        },
    },
}

TEST_GENERATION_EXAMPLES = {
    "minimal": {
        "summary": "Minimal (domain + required description)",
        "description": "Smallest valid request body.",
        "value": {
            "domain": "telecom_billing",
            "context": {
                "description": "Telecom billing chatbot. Users ask about charges, roaming, discounts, and invoice issues. Bot should ask clarifying questions before making claims."
            },
        },
    },
    "with_language_and_intents": {
        "summary": "With language + number_of_intents",
        "description": "Generate 5 intent categories and output in English.",
        "value": {
            "domain": "telecom_billing",
            "context": {
                "description": "Telecom billing chatbot. Users ask about charges, roaming, proration, plan changes, and refunds. Ask clarifying questions first.",
                "language": "en",
                "number_of_intents": 5,
            },
        },
    },
    "with_user_defined_vars": {
        "summary": "With userDefinedVariables",
        "description": "Use variables to shape test cases (country, segment, channel).",
        "value": {
            "domain": "telecom_billing",
            "context": {
                "description": "Telecom billing chatbot. Focus on postpaid Irish customers. Encourage the bot to ask for invoice details before resolving.",
                "language": "en",
                "number_of_intents": 5,
                "userDefinedVariables": {
                    "country": "IE",
                    "segment": "postpaid",
                    "channel": "web_chat",
                    "currency": "EUR",
                },
            },
        },
    },
}


# =========================
# System prompts
# =========================

SYSTEM_PROMPT_OBJECTIVE = """You are a helpful assistant that improves an objective into clearer, testable defining objectives.

Input: You will receive a JSON payload containing:
  - objective: string
  - context: optional object with fields like persona, domain, instructions, satisfactionCriteria, extraNotes, userVariables
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
- The first non-whitespace character MUST be '{' and the last MUST be '}'.
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
