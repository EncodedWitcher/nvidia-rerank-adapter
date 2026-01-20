"""
Embeddings API Router.
Provides transparent proxy to NVIDIA embeddings endpoint.
"""

import logging
from typing import Optional, Any, Dict

from fastapi import APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse

from services.nvidia_client import nvidia_client
from services.key_manager import key_manager
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Embeddings"])


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
    "/v1/embeddings",
    summary="Create embeddings",
    description="""
    Create embeddings for the given input text.
    
    This endpoint transparently proxies requests to NVIDIA's embeddings API.
    The request body is forwarded as-is, with automatic:
    - API key rotation (round-robin)
    - 429 rate limit retry handling
    - Model fallback if the requested model is not found
    """
)
async def create_embeddings(
    request: Request,
    _: None = Depends(verify_api_key)
) -> JSONResponse:
    """
    Create embeddings for input text.
    
    Transparently forwards the request to NVIDIA API with retry logic.
    
    Args:
        request: The raw HTTP request
        
    Returns:
        The NVIDIA API response
    """
    # Check if we have API keys available
    if key_manager.total_keys == 0:
        logger.error("No NVIDIA API keys configured")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: No API keys configured"
        )
    
    try:
        # Parse the request body
        request_body: Dict[str, Any] = await request.json()
        
        # Execute embeddings request with retry logic
        response = await nvidia_client.embeddings(request_body)
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception(f"Embeddings request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
