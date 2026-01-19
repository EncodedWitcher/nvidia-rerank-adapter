"""
Rerank API Router.
Provides OpenAI-compatible rerank endpoint.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from fastapi.responses import JSONResponse

from models import RerankRequest, RerankResponse, ErrorResponse, ErrorDetail
from services.nvidia_client import nvidia_client
from services.key_manager import key_manager
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Rerank"])


async def verify_api_key(
    authorization: Optional[str] = Header(None, description="Bearer token for authentication")
) -> None:
    """
    Verify the API key if authentication is enabled.
    
    Args:
        authorization: The Authorization header value
        
    Raises:
        HTTPException: If authentication fails
    """
    # Skip authentication if no API key is configured
    if not settings.api_key:
        return
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    # Extract token from "Bearer <token>" format
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Use 'Bearer <token>'"
        )
    
    token = parts[1]
    if token != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


@router.post(
    "/v1/rerank",
    response_model=RerankResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
    summary="Rerank documents",
    description="""
    Rerank a list of documents based on their relevance to a query.
    
    This endpoint is compatible with OpenAI's rerank API format (used by OpenWebUI).
    It converts the request to NVIDIA's format internally and returns results in the standard format.
    """
)
async def rerank(
    request: RerankRequest,
    _: None = Depends(verify_api_key)
) -> RerankResponse:
    """
    Rerank documents based on relevance to query.
    
    Args:
        request: The rerank request containing query and documents
        
    Returns:
        Reranked results with relevance scores
    """
    # Validate request
    if not request.documents:
        raise HTTPException(
            status_code=400,
            detail="At least one document is required"
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    # Check if we have API keys available
    if key_manager.total_keys == 0:
        logger.error("No NVIDIA API keys configured")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: No API keys configured"
        )
    
    if key_manager.active_keys == 0:
        logger.warning("All API keys are on cooldown")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable: All API keys are on cooldown"
        )
    
    try:
        # Execute rerank request
        response = await nvidia_client.rerank(request)
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception(f"Rerank request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/v1/models",
    summary="List available models",
    description="Get a list of available rerank models"
)
async def list_models() -> dict:
    """
    List available rerank models.
    
    Returns:
        List of available model configurations
    """
    from config import MODEL_CONFIGS
    
    models = [
        {
            "id": key,
            "object": "model",
            "owned_by": "nvidia",
            "nvidia_model": config["model"]
        }
        for key, config in MODEL_CONFIGS.items()
    ]
    
    return {
        "object": "list",
        "data": models
    }


@router.get(
    "/v1/keys/status",
    summary="Get API key status",
    description="Get the current status of API keys (for monitoring)"
)
async def get_key_status(
    _: None = Depends(verify_api_key)
) -> dict:
    """
    Get API key pool status.
    
    Returns:
        Status information about the API key pool
    """
    return key_manager.get_status()
