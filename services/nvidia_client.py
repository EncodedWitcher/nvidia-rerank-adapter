"""
NVIDIA API Client for Rerank and Embeddings Adapter.
Handles format conversion and API communication with NVIDIA.
"""

import asyncio
import math
import logging
from typing import List, Optional, Any, Dict
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
    Client for interacting with NVIDIA Rerank and Embeddings API.
    
    Handles:
    - Format conversion between OpenAI and NVIDIA formats
    - API key rotation with round-robin
    - 429 rate limit handling with retry
    - Model fallback for embeddings
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the NVIDIA client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
    
    async def _make_request_with_retry(
        self,
        url: str,
        payload: Dict[str, Any],
        fallback_model: Optional[str] = None
    ) -> httpx.Response:
        """
        Make HTTP request with key rotation and 429 retry logic.
        
        Args:
            url: The API endpoint URL
            payload: The request payload (will be sent as JSON)
            fallback_model: Optional fallback model to use if model not found
            
        Returns:
            The successful HTTP response
            
        Raises:
            ValueError: If no API keys are configured
            Exception: If the request fails after all retries
        """
        if key_manager.total_keys == 0:
            raise ValueError("No API keys available. Please configure NVIDIA_API_KEYS.")
        
        consecutive_429_count = 0
        last_error: Optional[Exception] = None
        
        while True:
            # Get next available API key
            api_key = key_manager.get_next_key()
            if not api_key:
                raise ValueError("No API keys available. Please configure NVIDIA_API_KEYS.")
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                    )
                    
                    # Handle 429 rate limit
                    if response.status_code == 429:
                        consecutive_429_count += 1
                        logger.warning(
                            f"Rate limited (429) - consecutive count: {consecutive_429_count}"
                        )
                        
                        # If all keys have been tried, wait before next round
                        if consecutive_429_count >= key_manager.total_keys:
                            logger.warning(
                                f"All {key_manager.total_keys} keys rate limited. "
                                f"Waiting {settings.retry_wait_seconds}s before retry..."
                            )
                            await asyncio.sleep(settings.retry_wait_seconds)
                            consecutive_429_count = 0
                        continue
                    
                    # Reset 429 counter on non-429 response
                    consecutive_429_count = 0
                    
                    # Handle model not found - retry with fallback model
                    if response.status_code == 404 and fallback_model:
                        error_text = response.text.lower()
                        if "model" in error_text or "not found" in error_text:
                            logger.warning(
                                f"Model not found, retrying with fallback model: {fallback_model}"
                            )
                            payload["model"] = fallback_model
                            # Retry with fallback model
                            response = await client.post(
                                url,
                                json=payload,
                                headers={
                                    "Authorization": f"Bearer {api_key}",
                                    "Content-Type": "application/json",
                                    "Accept": "application/json"
                                }
                            )
                            # If still 429, continue the loop
                            if response.status_code == 429:
                                consecutive_429_count += 1
                                continue
                    
                    # Handle authentication error
                    if response.status_code == 401:
                        logger.error(f"Authentication failed for key")
                        last_error = Exception(f"Authentication failed: {response.text}")
                        # Try next key
                        continue
                    
                    # Handle server errors - retry with next key
                    if response.status_code >= 500:
                        logger.error(f"NVIDIA server error: {response.text}")
                        last_error = Exception(f"Server error: {response.text}")
                        continue
                    
                    # Raise for other HTTP errors
                    response.raise_for_status()
                    
                    return response
                    
            except httpx.TimeoutException as e:
                logger.error(f"Request timeout: {e}")
                last_error = e
                # Continue to try next key
            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                last_error = e
                # Continue to try next key
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
    
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
        
        # Make request with retry logic
        response = await self._make_request_with_retry(
            url=nvidia_url,
            payload=nvidia_request.model_dump()
        )
        
        # Parse response
        nvidia_response = NvidiaRerankResponse.model_validate(response.json())
        
        # Convert and return response
        return self.convert_from_nvidia_format(nvidia_response, request.top_n)
    
    async def embeddings(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an embeddings request against NVIDIA API.
        Transparently forwards the request and handles 429 retry and model fallback.
        
        Args:
            request_body: The raw request body to forward
            
        Returns:
            The NVIDIA API response as a dictionary
            
        Raises:
            Exception: If the request fails after all retries
        """
        # Make request with retry logic and model fallback
        response = await self._make_request_with_retry(
            url=settings.nvidia_embeddings_url,
            payload=request_body,
            fallback_model=settings.default_embedding_model
        )
        
        return response.json()


# Global client instance
nvidia_client = NvidiaClient(timeout=settings.request_timeout)
