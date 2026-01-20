"""Routers package for NVIDIA Rerank Adapter."""

from .rerank import router as rerank_router
from .embeddings import router as embeddings_router

__all__ = ["rerank_router", "embeddings_router"]
