from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from src.core.config import Config
from src.core.bedrock_client import BedrockClient as CognitoBedrockClient
from src.local.bedrock_client import BedrockClient as LocalBedrockClient

from src.inference.recommendation import recommend_objective
from src.models.recommendation import SimpleObjectiveRequest, SimpleRecommendResponse

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

app = FastAPI(title="Cyara Recommendation Engine", version="1.0.0")

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
    response_model_exclude_none=True,  # ✅ omits "reason" when includeReason=false
    summary="Recommend clearer defining objective(s)",
    description="Takes a vague objective and optional context and returns clearer, testable defining objective(s).",
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
