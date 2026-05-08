"""
pymrsf.cache — Caching layer for RAG chunk scores

Avoids re-scoring the same chunks across multiple queries.
Thread-safe, configurable LRU cache with statistics.
"""

import hashlib
import time
from typing import Optional, Dict, Any
from functools import lru_cache
import threading


# ── Cache configuration ────────────────────────────────────────────────────────

_CACHE_ENABLED = True
_CACHE_MAX_SIZE = 10000  # Maximum number of cached scores
_CACHE_TTL = 3600  # Time-to-live in seconds (1 hour default)

# Global cache and statistics
_cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
_cache_lock = threading.Lock()
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,
}


def configure_cache(enabled: bool = True, max_size: int = 10000, ttl: int = 3600):
    """
    Configure the RAG score cache.

    Args:
        enabled: Enable/disable caching
        max_size: Maximum number of cached entries
        ttl: Time-to-live in seconds (0 = no expiration)
    """
    global _CACHE_ENABLED, _CACHE_MAX_SIZE, _CACHE_TTL
    _CACHE_ENABLED = enabled
    _CACHE_MAX_SIZE = max_size
    _CACHE_TTL = ttl


def _make_cache_key(chunk: str, query: Optional[str], weights: Optional[dict]) -> str:
    """
    Generate a deterministic cache key from chunk, query, and weights.
    
    Args:
        chunk: The text chunk
        query: The query (optional)
        weights: Scoring weights (optional)
    
    Returns:
        SHA256 hash as cache key
    """
    # Normalize weights to ensure consistent hashing
    if weights:
        weights_str = str(sorted(weights.items()))
    else:
        weights_str = ""
    
    cache_input = f"{chunk}|{query or ''}|{weights_str}"
    return hashlib.sha256(cache_input.encode()).hexdigest()


def get_cached_score(chunk: str, query: Optional[str] = None, weights: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached score if available and not expired.
    
    Args:
        chunk: The text chunk
        query: The query (optional)
        weights: Scoring weights (optional)
    
    Returns:
        Cached result dict or None if not found/expired
    """
    if not _CACHE_ENABLED:
        return None
    
    key = _make_cache_key(chunk, query, weights)
    
    with _cache_lock:
        if key in _cache:
            result, timestamp = _cache[key]
            
            # Check TTL
            if _CACHE_TTL > 0 and (time.time() - timestamp) > _CACHE_TTL:
                # Expired - remove it
                del _cache[key]
                _cache_stats["misses"] += 1
                return None
            
            _cache_stats["hits"] += 1
            return result.copy()  # Return a copy to avoid mutation
        
        _cache_stats["misses"] += 1
        return None


def set_cached_score(chunk: str, query: Optional[str], weights: Optional[dict], result: Dict[str, Any]):
    """
    Store a score result in the cache.
    
    Args:
        chunk: The text chunk
        query: The query (optional)
        weights: Scoring weights (optional)
        result: The scoring result to cache
    """
    if not _CACHE_ENABLED:
        return
    
    key = _make_cache_key(chunk, query, weights)
    
    with _cache_lock:
        # LRU eviction if cache is full
        if len(_cache) >= _CACHE_MAX_SIZE:
            # Evict oldest entry
            oldest_key = min(_cache.keys(), key=lambda k: _cache[k][1])
            del _cache[oldest_key]
            _cache_stats["evictions"] += 1
        
        _cache[key] = (result.copy(), time.time())


def clear_cache():
    """Clear all cached scores."""
    with _cache_lock:
        _cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with hits, misses, evictions, size, and hit_rate
    """
    with _cache_lock:
        total = _cache_stats["hits"] + _cache_stats["misses"]
        hit_rate = _cache_stats["hits"] / total if total > 0 else 0.0
        
        return {
            "hits": _cache_stats["hits"],
            "misses": _cache_stats["misses"],
            "evictions": _cache_stats["evictions"],
            "size": len(_cache),
            "max_size": _CACHE_MAX_SIZE,
            "hit_rate": round(hit_rate * 100, 2),
            "enabled": _CACHE_ENABLED,
        }


def reset_cache_stats():
    """Reset cache statistics counters."""
    with _cache_lock:
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0
        _cache_stats["evictions"] = 0


def print_cache_stats():
    """Print a formatted cache statistics report."""
    stats = get_cache_stats()
    
    print(f"\n{'═' * 60}")
    print(f"  PYMRSF CACHE STATISTICS")
    print(f"{'═' * 60}")
    print(f"  Status      : {'Enabled' if stats['enabled'] else 'Disabled'}")
    print(f"  Cache size  : {stats['size']:,} / {stats['max_size']:,}")
    print(f"  Hits        : {stats['hits']:,}")
    print(f"  Misses      : {stats['misses']:,}")
    print(f"  Evictions   : {stats['evictions']:,}")
    print(f"  Hit rate    : {stats['hit_rate']:.1f}%")
    print(f"{'═' * 60}\n")


# ── Embedding cache (for relevance scoring) ────────────────────────────────────

@lru_cache(maxsize=1000)
def _cached_text_hash(text: str) -> str:
    """Cache text hashes for embedding lookups."""
    return hashlib.sha256(text.encode()).hexdigest()


_embedding_cache: Dict[str, Any] = {}
_embedding_lock = threading.Lock()


def get_cached_embedding(text: str):
    """Get cached embedding for text."""
    if not _CACHE_ENABLED:
        return None
    
    text_hash = _cached_text_hash(text)
    with _embedding_lock:
        return _embedding_cache.get(text_hash)


def set_cached_embedding(text: str, embedding):
    """Cache an embedding for text."""
    if not _CACHE_ENABLED:
        return
    
    text_hash = _cached_text_hash(text)
    with _embedding_lock:
        # Simple LRU: if cache is too big, clear it
        if len(_embedding_cache) >= 5000:
            _embedding_cache.clear()
        _embedding_cache[text_hash] = embedding


def clear_embedding_cache():
    """Clear the embedding cache."""
    with _embedding_lock:
        _embedding_cache.clear()
