from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from src.core.config import Config
from src.core.bedrock_client import BedrockClient as CognitoBedrockClient
from src.local.bedrock_client import BedrockClient as LocalBedrockClient

from src.inference.recommendation import recommend_objective
from src.inference.test_generation import generate_test_cases

from src.models.recommendation import SimpleObjectiveRequest, SimpleRecommendResponse
from src.models.test_generation import TestGenerationRequest, TestGenerationResponse

config = Config.load_config()

# Standard: ENV is lowercase everywhere
env = (config.get("env") or "dev").strip().lower()

# Always use Cognito unless local
if env == "local":
    print("Using LOCAL mock Bedrock client (env=local)")
    bedrock_client = LocalBedrockClient(
        region_name=config["region"],
        endpoint_url=config.get("aws_endpoint"),
    )
else:
    print(f"Using COGNITO Bedrock client (env={env})")
    bedrock_client = CognitoBedrockClient(
        region_name=config["region"],
        config=config,
        endpoint_url=config.get("aws_endpoint"),
    )

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

app = FastAPI(
    title="Objective Recommendation API",
    version="2.0.0",
    description=API_DESCRIPTION,
)

api_key_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    scheme_name="ApiKeyAuth",
)


def verify_api_key(api_key: str | None):
    expected = config.get("api_key")
    if not expected:
        raise HTTPException(status_code=500, detail="API_KEY is not configured")
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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

**Authentication:** requires `X-API-Key` header.
"""


@app.post(
    f"/{env}/recommendation",
    response_model=SimpleRecommendResponse,
    response_model_exclude_none=True,  # omits reason when includeReason=false
    summary="Recommend clearer defining objective",
    description=RECOMMENDATION_DESC,
)
async def handle_recommendation(
    req: SimpleObjectiveRequest,
    api_key: str | None = Security(api_key_scheme),
):
    verify_api_key(api_key)

    model_id = config.get("bedrock_model_id")
    if not model_id:
        raise HTTPException(status_code=500, detail="BEDROCK_MODEL_ID is not configured")

    try:
        return recommend_objective(req, bedrock_client=bedrock_client, model_id=model_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


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


@app.post(
    f"/{env}/test-generation",
    response_model=TestGenerationResponse,
    summary="Generate test cases",
    description=TEST_GEN_DESC,
)
async def handle_test_generation(
    req: TestGenerationRequest,
    api_key: str | None = Security(api_key_scheme),
):
    verify_api_key(api_key)

    model_id = config.get("bedrock_model_id")
    if not model_id:
        raise HTTPException(status_code=500, detail="BEDROCK_MODEL_ID is not configured")

    try:
        return generate_test_cases(req, bedrock_client=bedrock_client, model_id=model_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
