# src/eval/cache.py
"""
Caching for LLM-as-judge results.

Reduces cost by caching identical evaluations.
Cache key = hash(question + answer + context)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class LLMJudgeCache:
    """
    Simple file-based cache for LLM judge results.
    
    Cache is stored per-scorer type to allow independent invalidation.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "runs" / "cache"
        self.cache_dir = cache_dir
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, scorer_name: str, **inputs: str) -> str:
        """Generate deterministic cache key from inputs."""
        # Sort keys for deterministic ordering
        content = "|".join(f"{k}={v}" for k, v in sorted(inputs.items()))
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{scorer_name}_{hash_val}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, scorer_name: str, **inputs: str) -> Optional[dict[str, Any]]:
        """
        Get cached result if exists.
        
        Args:
            scorer_name: Name of the scorer (e.g., "faithfulness")
            **inputs: The inputs that determine the cache key
        
        Returns:
            Cached result dict or None if not cached
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(scorer_name, **inputs)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupted cache, ignore
                return None
        
        return None
    
    def set(self, scorer_name: str, result: dict[str, Any], **inputs: str) -> None:
        """
        Cache a result.
        
        Args:
            scorer_name: Name of the scorer
            result: The result to cache
            **inputs: The inputs that determine the cache key
        """
        if not self.enabled:
            return
        
        cache_key = self._get_cache_key(scorer_name, **inputs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, "w") as f:
                json.dump(result, f)
        except IOError:
            # Failed to write cache, ignore
            pass
    
    def clear(self, scorer_name: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            scorer_name: If provided, only clear entries for this scorer.
                        If None, clear all entries.
        
        Returns:
            Number of entries cleared
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        pattern = f"{scorer_name}_*.json" if scorer_name else "*.json"
        
        for cache_file in self.cache_dir.glob(pattern):
            try:
                cache_file.unlink()
                count += 1
            except IOError:
                pass
        
        return count
    
    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {"total": 0, "size_bytes": 0}
        
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files if f.exists())
        
        return {
            "total": len(files),
            "size_bytes": total_size,
        }


# Global cache instance
_cache: Optional[LLMJudgeCache] = None


def get_cache(enabled: bool = True) -> LLMJudgeCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = LLMJudgeCache(enabled=enabled)
    return _cache
