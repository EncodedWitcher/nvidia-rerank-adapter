"""
Configuration module for NVIDIA Rerank Adapter.
Handles API keys, model configurations, and environment variables.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        # NVIDIA API Keys (comma-separated)
        self.nvidia_api_keys: List[str] = self._parse_api_keys()
        
        # Default model to use if not specified in request
        self.default_model: str = os.getenv("DEFAULT_MODEL", "nvidia/rerank-qa-mistral-4b")
        
        # Server configuration
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        
        # Request timeout in seconds
        self.request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # API Key for authentication (optional)
        self.api_key: Optional[str] = os.getenv("API_KEY", None)
    
    def _parse_api_keys(self) -> List[str]:
        """Parse comma-separated API keys from environment variable."""
        keys_str = os.getenv("NVIDIA_API_KEYS", "")
        if not keys_str:
            return []
        return [key.strip() for key in keys_str.split(",") if key.strip()]


# Model configuration mapping
# Maps model aliases to NVIDIA API endpoints and model names
# This allows easy extensibility for different NVIDIA rerank models
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    # Default reranker alias (for compatibility)
    "reranker": {
        "url": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        "model": "nvidia/rerank-qa-mistral-4b"
    },
    # NVIDIA Rerank QA Mistral 4B
    "nvidia/rerank-qa-mistral-4b": {
        "url": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        "model": "nvidia/rerank-qa-mistral-4b"
    },
    # NVIDIA NV Rerank QA Mistral 4B (alternative)
    "nvidia/nv-rerank-qa-mistral-4b": {
        "url": "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerank-qa-mistral-4b/reranking",
        "model": "nvidia/nv-rerank-qa-mistral-4b"
    },
    # NVIDIA Llama3 Reranker
    "nvidia/llama-3.2-nv-rerankqa-1b-v2": {
        "url": "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3.2-nv-rerankqa-1b-v2/reranking",
        "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    },
    # Add more models as needed...
}


def get_model_config(model_name: str) -> Optional[Dict[str, str]]:
    """
    Get model configuration by name.
    
    Args:
        model_name: The model name or alias
        
    Returns:
        Model configuration dict with 'url' and 'model' keys, or None if not found
    """
    # Direct match
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Try lowercase match
    model_lower = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if key.lower() == model_lower:
            return config
    
    # If not found, return default configuration with the provided model name
    # This allows dynamic model specification
    return {
        "url": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        "model": model_name
    }


# Global settings instance
settings = Settings()
