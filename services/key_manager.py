"""
API Key Manager for NVIDIA Rerank Adapter.
Implements simple round-robin key rotation.
"""

import threading
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class KeyManager:
    """
    Manages multiple API keys with round-robin rotation.
    
    Features:
    - Simple round-robin key selection
    - Thread-safe operations
    """
    
    def __init__(self, api_keys: List[str]):
        """
        Initialize the key manager.
        
        Args:
            api_keys: List of API keys to manage
        """
        self._lock = threading.Lock()
        self._keys: List[str] = list(api_keys)
        self._current_index: int = 0
    
    @property
    def total_keys(self) -> int:
        """Get total number of keys."""
        return len(self._keys)
    
    @property
    def active_keys(self) -> int:
        """Get number of currently active keys (same as total for simple rotation)."""
        return len(self._keys)
    
    def add_key(self, key: str) -> None:
        """Add a new API key to the pool."""
        with self._lock:
            # Check if key already exists
            if key in self._keys:
                logger.warning(f"Key already exists in pool, skipping...")
                return
            self._keys.append(key)
            logger.info(f"Added new key to pool. Total keys: {len(self._keys)}")
    
    def remove_key(self, key: str) -> bool:
        """Remove an API key from the pool."""
        with self._lock:
            if key in self._keys:
                self._keys.remove(key)
                # Adjust current index if necessary
                if self._current_index >= len(self._keys):
                    self._current_index = 0
                logger.info(f"Removed key from pool. Total keys: {len(self._keys)}")
                return True
            return False
    
    def get_next_key(self) -> Optional[str]:
        """
        Get the next available API key using round-robin.
        
        Returns:
            The next available API key, or None if no keys are available
        """
        with self._lock:
            if not self._keys:
                logger.error("No API keys configured")
                return None
            
            key = self._keys[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._keys)
            return key
    
    def get_status(self) -> dict:
        """Get the current status of all keys."""
        with self._lock:
            return {
                "total_keys": len(self._keys),
                "active_keys": len(self._keys),
                "keys": [
                    {
                        "key_prefix": k[:10] + "..." if len(k) > 10 else k,
                    }
                    for k in self._keys
                ]
            }


# Import settings here to avoid circular imports
from config import settings

# Global key manager instance
key_manager = KeyManager(api_keys=settings.nvidia_api_keys)
