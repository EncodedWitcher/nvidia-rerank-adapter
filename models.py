"""
Pydantic models for request/response validation.
Defines both OpenAI-compatible and NVIDIA-specific data structures.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================
# OpenAI/OpenWebUI Compatible Models (Input/Output)
# ============================================

class RerankRequest(BaseModel):
    """
    OpenAI-compatible rerank request model.
    This is the format that clients (like OpenWebUI) will send.
    """
    model: str = Field(
        default="reranker",
        description="Model name or alias to use for reranking"
    )
    query: str = Field(
        ...,
        description="The query text to compare documents against"
    )
    documents: List[str] = Field(
        ...,
        description="List of document texts to rerank"
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Number of top results to return. If None, returns all documents."
    )


class RerankResult(BaseModel):
    """Single rerank result item."""
    index: int = Field(
        ...,
        description="Original index of the document in the input list"
    )
    relevance_score: float = Field(
        ...,
        description="Relevance score between 0 and 1"
    )


class RerankResponse(BaseModel):
    """
    OpenAI-compatible rerank response model.
    This is the format that will be returned to clients.
    """
    results: List[RerankResult] = Field(
        ...,
        description="List of reranked results sorted by relevance score (descending)"
    )


# ============================================
# NVIDIA API Models (Internal)
# ============================================

class NvidiaQueryText(BaseModel):
    """NVIDIA query text wrapper."""
    text: str


class NvidiaPassage(BaseModel):
    """NVIDIA passage/document wrapper."""
    text: str


class NvidiaRerankRequest(BaseModel):
    """
    NVIDIA API rerank request format.
    Used internally to communicate with NVIDIA API.
    """
    query: NvidiaQueryText
    passages: List[NvidiaPassage]
    model: str
    truncate: str = "NONE"


class NvidiaRankingItem(BaseModel):
    """NVIDIA ranking result item."""
    index: int
    logit: float


class NvidiaRerankResponse(BaseModel):
    """
    NVIDIA API rerank response format.
    Used internally to parse NVIDIA API responses.
    """
    rankings: List[NvidiaRankingItem]


# ============================================
# Error Models
# ============================================

class ErrorDetail(BaseModel):
    """Error detail model."""
    message: str
    type: str = "error"
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: ErrorDetail
