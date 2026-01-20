"""
NVIDIA Rerank Adapter - Main Application Entry Point

A FastAPI-based adapter that converts OpenAI-compatible rerank requests
to NVIDIA's rerank API format.

Features:
- OpenAI/OpenWebUI compatible rerank endpoint
- Multiple API key rotation with round-robin
- Automatic failover and cooldown handling
- Extensible model configuration
"""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import rerank_router, embeddings_router
from services.key_manager import key_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 50)
    logger.info("NVIDIA Rerank Adapter Starting...")
    logger.info("=" * 50)
    
    # Log configuration
    logger.info(f"Server: {settings.host}:{settings.port}")
    logger.info(f"Default Model: {settings.default_model}")
    logger.info(f"Request Timeout: {settings.request_timeout}s")
    logger.info(f"API Keys Configured: {key_manager.total_keys}")
    logger.info(f"Authentication: {'Enabled' if settings.api_key else 'Disabled'}")
    
    if key_manager.total_keys == 0:
        logger.warning(
            "⚠️  No NVIDIA API keys configured! "
            "Set NVIDIA_API_KEYS environment variable."
        )
    
    logger.info("=" * 50)
    logger.info("Server ready to accept requests")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("NVIDIA Rerank Adapter Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="NVIDIA Rerank Adapter",
    description="""
    A proxy adapter that converts OpenAI-compatible rerank API requests 
    to NVIDIA's rerank API format.
    
    ## Features
    
    - **OpenAI Compatible**: Works with OpenWebUI and other OpenAI-compatible clients
    - **Multi-Key Support**: Rotate between multiple NVIDIA API keys
    - **Auto Failover**: Automatically switch keys on failure
    - **Extensible**: Easy to add new NVIDIA rerank models
    
    ## Usage
    
    Send POST requests to `/v1/rerank` with the following format:
    
    ```json
    {
        "model": "reranker",
        "query": "Your search query",
        "documents": ["Document 1", "Document 2", "Document 3"],
        "top_n": 3
    }
    ```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rerank_router)
app.include_router(embeddings_router)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "NVIDIA Rerank Adapter",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "rerank": "/v1/rerank",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "api_keys": {
            "total": key_manager.total_keys,
            "active": key_manager.active_keys
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info"
    )
