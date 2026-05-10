import os
from dataclasses import dataclass, field

# Experimental research backend — re-exported at top level for backward compat
from .experimental import (
    mrsf_write, mrsf_read, mrsf_read_novel, mrsf_delete, rebuild_index,
    save_index, load_index, reset_index_metadata, close_connections,
    mrsf_inspect, mrsf_rebuild_explained,
    mrsf_benchmark_canterbury, mrsf_latency_benchmark,
)
from . import experimental

# Backward compatibility alias (deprecated, will be removed in v0.6)
rebuild_faiss_from_sqlite = reset_index_metadata
from .embeddings import embed, get_embedding_dim
from .probe import probe, probe_compare
from .rag import (
    score_chunk, score_chunks, score_chunks_batch,
    explain_chunk, filter_chunks, smart_filter, DEFAULT_WEIGHTS,
    score_chunk_async, score_chunks_async, filter_chunks_async,
    DEFAULT_RELEVANCE_CUTOFF, DEFAULT_RAG_THRESHOLDS,
    WeightConfig,
)
from .chunker import smart_chunk
from .core import (
    ModelSession, compute_delta, get_surprises, tokenize, detokenize,
    quantized_argmax, next_token_greedy,
    get_backend, get_raw_lm, provider_capabilities, set_provider,
    PROVIDER, MODEL_VERSION, LOGIT_PRECISION,
)
from . import cache
from .cache import (
    configure_cache, get_cache_stats, get_embedding_cache_stats,
    reset_cache_stats, clear_cache, clear_embedding_cache
)

__version__ = "0.4.1"


# ── Centralized runtime configuration ─────────────────────────────────────────

@dataclass
class Config:
    """Central configuration for all pymrsf settings.

    Reads defaults from environment variables so existing .env setups
    continue to work. Override at runtime with configure().
    """
    provider: str = field(default_factory=lambda: os.getenv("PYMRSF_PROVIDER", "local"))
    model_path: str = field(default_factory=lambda: os.getenv("PYMRSF_MODEL_PATH", "./models/mistral-7b-v0.1.Q4_K_M.gguf"))
    n_ctx: int = field(default_factory=lambda: int(os.getenv("PYMRSF_N_CTX", "4096")))
    n_gpu_layers: int = field(default_factory=lambda: int(os.getenv("PYMRSF_N_GPU_LAYERS", "0")))
    ollama_base: str = field(default_factory=lambda: os.getenv("PYMRSF_OLLAMA_BASE", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: os.getenv("PYMRSF_EMBED_MODEL", "nomic-embed-text"))
    embed_timeout: int = field(default_factory=lambda: int(os.getenv("PYMRSF_EMBED_TIMEOUT", "30")))
    embed_dim: int = field(default_factory=lambda: int(os.getenv("PYMRSF_EMBED_DIM", "768")))
    logit_precision: int = field(default_factory=lambda: int(os.getenv("PYMRSF_LOGIT_PRECISION", "6")))
    db_path: str = field(default_factory=lambda: os.getenv("PYMRSF_DB_PATH", "mrsf.db"))
    faiss_path: str = field(default_factory=lambda: os.getenv("PYMRSF_FAISS_PATH", "mrsf.faiss"))
    default_relevance_cutoff: float = 0.30
    default_min_rag_score: int = 50
    default_diversity_threshold: float = 0.85


_config = Config()


def configure(**kwargs) -> Config:
    """Set global pymrsf configuration at runtime.

    Accepts any field name from Config. Individual modules continue
    to read os.getenv for now (backward compat with existing .env setups).

    Example:
        import pymrsf
        pymrsf.configure(provider="openai", default_relevance_cutoff=0.40)
    """
    global _config
    for k, v in kwargs.items():
        if hasattr(_config, k):
            setattr(_config, k, v)
        else:
            raise ValueError(f"Unknown config key: '{k}'. Valid keys: {list(_config.__dataclass_fields__)}")
    return _config


def get_config() -> Config:
    """Return the current global Config instance."""
    return _config

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    # Core functions
    "tokenize",
    "detokenize",
    "quantized_argmax",
    "get_surprises",
    "compute_delta",
    "next_token_greedy",
    "ModelSession",
    
    # Backend access
    "get_backend",
    "get_raw_lm",
    "provider_capabilities",
    "set_provider",
    
    # RAG scoring
    "score_chunk",
    "score_chunks",
    "score_chunks_batch",
    "explain_chunk",
    "filter_chunks",
    "smart_filter",
    "smart_chunk",
    "score_chunk_async",
    "score_chunks_async",
    "filter_chunks_async",
    "DEFAULT_WEIGHTS",
    "DEFAULT_RELEVANCE_CUTOFF",
    "DEFAULT_RAG_THRESHOLDS",
    "WeightConfig",
    
    # Knowledge probing
    "probe",
    "probe_compare",
    
    # Experimental: storage & compression
    "experimental",
    "mrsf_write",
    "mrsf_read",
    "mrsf_read_novel",
    "mrsf_delete",
    "rebuild_index",
    "save_index",
    "load_index",
    "reset_index_metadata",
    "rebuild_faiss_from_sqlite",  # Deprecated alias — removed in v0.6
    "close_connections",

    # Experimental: inspection & debugging
    "mrsf_inspect",
    "mrsf_rebuild_explained",

    # Experimental: benchmarking
    "mrsf_benchmark_canterbury",
    "mrsf_latency_benchmark",
    
    # Embeddings
    "embed",
    "get_embedding_dim",
    
    # Cache module
    "cache",
    
    # Cache configuration functions
    "configure_cache",
    "get_cache_stats",
    "get_embedding_cache_stats",
    "reset_cache_stats",
    "clear_cache",
    "clear_embedding_cache",
    
    # Constants
    "PROVIDER",
    "MODEL_VERSION",
    "LOGIT_PRECISION",
    
    # Runtime configuration
    "Config",
    "configure",
    "get_config",

    # Version
    "__version__",
]
