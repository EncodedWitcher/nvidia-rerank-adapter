"""
API Key Manager for NVIDIA Rerank Adapter.
Implements round-robin key rotation with failure handling.
"""

import threading
import time
import logging
from typing import List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KeyStatus:
    """Tracks the status of an API key."""
    key: str
    is_active: bool = True
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    cooldown_until: Optional[float] = None


class KeyManager:
    """
    Manages multiple API keys with round-robin rotation.
    
    Features:
    - Round-robin key selection
    - Automatic key cooldown on failure
    - Thread-safe operations
    - Key recovery after cooldown period
    """
    
    def __init__(
        self,
        api_keys: List[str],
        max_failures: int = 3,
        cooldown_seconds: int = 300
    ):
        """
        Initialize the key manager.
        
        Args:
            api_keys: List of API keys to manage
            max_failures: Number of failures before a key is put on cooldown
            cooldown_seconds: Seconds to wait before retrying a failed key
        """
        self._lock = threading.Lock()
        self._keys: List[KeyStatus] = [KeyStatus(key=key) for key in api_keys]
        self._current_index: int = 0
        self._max_failures = max_failures
        self._cooldown_seconds = cooldown_seconds
    
    @property
    def total_keys(self) -> int:
        """Get total number of keys."""
        return len(self._keys)
    
    @property
    def active_keys(self) -> int:
        """Get number of currently active keys."""
        self._check_cooldowns()
        with self._lock:
            return sum(1 for k in self._keys if k.is_active)
    
    def add_key(self, key: str) -> None:
        """Add a new API key to the pool."""
        with self._lock:
            # Check if key already exists
            if any(k.key == key for k in self._keys):
                logger.warning(f"Key already exists in pool, skipping...")
                return
            self._keys.append(KeyStatus(key=key))
            logger.info(f"Added new key to pool. Total keys: {len(self._keys)}")
    
    def remove_key(self, key: str) -> bool:
        """Remove an API key from the pool."""
        with self._lock:
            for i, k in enumerate(self._keys):
                if k.key == key:
                    self._keys.pop(i)
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
        self._check_cooldowns()
        
        with self._lock:
            if not self._keys:
                logger.error("No API keys configured")
                return None
            
            # Try to find an active key starting from current index
            attempts = 0
            while attempts < len(self._keys):
                key_status = self._keys[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._keys)
                
                if key_status.is_active:
                    return key_status.key
                
                attempts += 1
            
            logger.error("All API keys are on cooldown")
            return None
    
    def report_success(self, key: str) -> None:
        """Report successful use of a key (resets failure count)."""
        with self._lock:
            for key_status in self._keys:
                if key_status.key == key:
                    if key_status.failure_count > 0:
                        logger.info(f"Key recovered after {key_status.failure_count} failures")
                    key_status.failure_count = 0
                    key_status.is_active = True
                    key_status.cooldown_until = None
                    break
    
    def report_failure(self, key: str) -> None:
        """
        Report a failure for a key.
        If failures exceed threshold, put key on cooldown.
        """
        with self._lock:
            for key_status in self._keys:
                if key_status.key == key:
                    key_status.failure_count += 1
                    key_status.last_failure_time = time.time()
                    
                    if key_status.failure_count >= self._max_failures:
                        key_status.is_active = False
                        key_status.cooldown_until = time.time() + self._cooldown_seconds
                        logger.warning(
                            f"Key put on cooldown after {key_status.failure_count} failures. "
                            f"Will retry in {self._cooldown_seconds} seconds"
                        )
                    else:
                        logger.warning(
                            f"Key failure {key_status.failure_count}/{self._max_failures}"
                        )
                    break
    
    def _check_cooldowns(self) -> None:
        """Check and recover keys that have completed their cooldown period."""
        current_time = time.time()
        
        with self._lock:
            for key_status in self._keys:
                if (
                    not key_status.is_active
                    and key_status.cooldown_until is not None
                    and current_time >= key_status.cooldown_until
                ):
                    key_status.is_active = True
                    key_status.failure_count = 0
                    key_status.cooldown_until = None
                    logger.info("Key recovered from cooldown")
    
    def get_status(self) -> dict:
        """Get the current status of all keys."""
        self._check_cooldowns()
        
        with self._lock:
            return {
                "total_keys": len(self._keys),
                "active_keys": sum(1 for k in self._keys if k.is_active),
                "keys": [
                    {
                        "key_prefix": k.key[:10] + "..." if len(k.key) > 10 else k.key,
                        "is_active": k.is_active,
                        "failure_count": k.failure_count,
                        "cooldown_remaining": (
                            max(0, int(k.cooldown_until - time.time()))
                            if k.cooldown_until else None
                        )
                    }
                    for k in self._keys
                ]
            }


# Import settings here to avoid circular imports
from config import settings

# Global key manager instance
key_manager = KeyManager(
    api_keys=settings.nvidia_api_keys,
    max_failures=3,
    cooldown_seconds=300
)
