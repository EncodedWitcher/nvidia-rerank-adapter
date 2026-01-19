"""
NVIDIA API Client for Rerank Adapter.
Handles format conversion and API communication with NVIDIA.
"""

import math
import logging
from typing import List, Optional
import httpx

from models import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    NvidiaRerankRequest,
    NvidiaQueryText,
    NvidiaPassage,
    NvidiaRerankResponse,
)
from config import settings, get_model_config
from .key_manager import key_manager

logger = logging.getLogger(__name__)


class NvidiaClient:
    """
    Client for interacting with NVIDIA Rerank API.
    
    Handles:
    - Format conversion between OpenAI and NVIDIA formats
    - API key rotation
    - Error handling and retries
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the NVIDIA client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
    
    @staticmethod
    def convert_to_nvidia_format(
        request: RerankRequest,
        nvidia_model: str
    ) -> NvidiaRerankRequest:
        """
        Convert OpenAI-compatible request to NVIDIA format.
        
        Args:
            request: The incoming OpenAI-format request
            nvidia_model: The NVIDIA model identifier
            
        Returns:
            NVIDIA-format request object
        """
        return NvidiaRerankRequest(
            query=NvidiaQueryText(text=request.query),
            passages=[NvidiaPassage(text=doc) for doc in request.documents],
            model=nvidia_model,
            truncate="NONE"
        )
    
    @staticmethod
    def logit_to_score(logit: float) -> float:
        """
        Convert NVIDIA logit to relevance score (0-1).
        
        Uses sigmoid function: score = 1 / (1 + exp(-logit))
        
        Args:
            logit: The raw logit value from NVIDIA API
            
        Returns:
            Normalized score between 0 and 1
        """
        try:
            # Clamp logit to prevent overflow
            logit = max(-500, min(500, logit))
            return 1.0 / (1.0 + math.exp(-logit))
        except OverflowError:
            return 0.0 if logit < 0 else 1.0
    
    @staticmethod
    def convert_from_nvidia_format(
        nvidia_response: NvidiaRerankResponse,
        top_n: Optional[int] = None
    ) -> RerankResponse:
        """
        Convert NVIDIA response to OpenAI-compatible format.
        
        Args:
            nvidia_response: The NVIDIA API response
            top_n: Optional limit on number of results to return
            
        Returns:
            OpenAI-compatible response object
        """
        # Convert and sort results by relevance score (descending)
        results = [
            RerankResult(
                index=item.index,
                relevance_score=NvidiaClient.logit_to_score(item.logit)
            )
            for item in nvidia_response.rankings
        ]
        
        # Sort by relevance score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply top_n limit if specified
        if top_n is not None and top_n > 0:
            results = results[:top_n]
        
        return RerankResponse(results=results)
    
    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """
        Execute a rerank request against NVIDIA API.
        
        Args:
            request: The OpenAI-compatible rerank request
            
        Returns:
            OpenAI-compatible rerank response
            
        Raises:
            Exception: If the request fails after all retries
        """
        # Get model configuration
        model_config = get_model_config(request.model)
        if not model_config:
            raise ValueError(f"Unknown model: {request.model}")
        
        nvidia_url = model_config["url"]
        nvidia_model = model_config["model"]
        
        # Convert request format
        nvidia_request = self.convert_to_nvidia_format(request, nvidia_model)
        
        # Try with different keys on failure
        last_error: Optional[Exception] = None
        max_retries = min(3, key_manager.total_keys) if key_manager.total_keys > 0 else 1
        
        for attempt in range(max_retries):
            # Get next available API key
            api_key = key_manager.get_next_key()
            if not api_key:
                raise ValueError("No API keys available. Please configure NVIDIA_API_KEYS.")
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        nvidia_url,
                        json=nvidia_request.model_dump(),
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                    )
                    
                    # Check for HTTP errors
                    if response.status_code == 401:
                        logger.error(f"Authentication failed for key (attempt {attempt + 1})")
                        key_manager.report_failure(api_key)
                        last_error = Exception(f"Authentication failed: {response.text}")
                        continue
                    
                    if response.status_code == 429:
                        logger.warning(f"Rate limited (attempt {attempt + 1})")
                        key_manager.report_failure(api_key)
                        last_error = Exception(f"Rate limited: {response.text}")
                        continue
                    
                    if response.status_code >= 500:
                        logger.error(f"NVIDIA server error (attempt {attempt + 1}): {response.text}")
                        last_error = Exception(f"Server error: {response.text}")
                        continue
                    
                    response.raise_for_status()
                    
                    # Parse response
                    nvidia_response = NvidiaRerankResponse.model_validate(response.json())
                    
                    # Report success and reset failure count
                    key_manager.report_success(api_key)
                    
                    # Convert and return response
                    return self.convert_from_nvidia_format(nvidia_response, request.top_n)
                    
            except httpx.TimeoutException as e:
                logger.error(f"Request timeout (attempt {attempt + 1}): {e}")
                key_manager.report_failure(api_key)
                last_error = e
            except httpx.RequestError as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                key_manager.report_failure(api_key)
                last_error = e
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                last_error = e
                # Don't report failure for non-API errors
                raise
        
        # All retries failed
        raise Exception(f"All API requests failed. Last error: {last_error}")


# Global client instance
nvidia_client = NvidiaClient(timeout=settings.request_timeout)
