from fastapi import FastAPI, HTTPException, Security, Body
from fastapi.security.api_key import APIKeyHeader

from src.core.config import Config
from src.core.bedrock_client import BedrockClient as CognitoBedrockClient
from src.local.bedrock_client import BedrockClient as LocalBedrockClient

from src.inference.rec_objective import recommend_objective
from src.inference.rec_test_generation import generate_test_cases

from src.models.recommendation import SimpleObjectiveRequest, SimpleRecommendResponse
from src.models.test_generation import TestGenerationRequest, TestGenerationResponse

from src.docs.descriptions import (
    API_DESCRIPTION,
    RECOMMENDATION_DESC,
    TEST_GEN_DESC,
    RECOMMENDATION_EXAMPLES,
    TEST_GENERATION_EXAMPLES,
)

config = Config.load_config()

# Enforce env exists and is correct
env = config.get("env")
if not env:
    raise RuntimeError("Missing required config key: env (set ENV via env vars or Secrets Manager).")
if env != env.lower():
    raise RuntimeError(f"ENV must be lowercase (got: {env!r}). Expected e.g. 'local', 'dev', 'prod'.")

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


@app.post(
    f"/{env}/recommendation",
    response_model=SimpleRecommendResponse,
    response_model_exclude_none=True,
    summary="Recommend clearer defining objective",
    description=RECOMMENDATION_DESC,
)
async def handle_recommendation(
    req: SimpleObjectiveRequest = Body(..., openapi_examples=RECOMMENDATION_EXAMPLES),
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


@app.post(
    f"/{env}/test-generation",
    response_model=TestGenerationResponse,
    summary="Generate test cases",
    description=TEST_GEN_DESC,
)
async def handle_test_generation(
    req: TestGenerationRequest = Body(..., openapi_examples=TEST_GENERATION_EXAMPLES),
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
