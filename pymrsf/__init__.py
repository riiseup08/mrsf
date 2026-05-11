"""
pymrsf — Model-Relative Semantic Filtering

Split text at knowledge boundaries and score RAG chunks by information gain:

    from pymrsf import smart_chunk, score_chunk, filter_chunks, probe

    # Chunk at natural topic boundaries using model surprise
    pieces = smart_chunk(long_document)

    # Score a candidate chunk against a query
    result = score_chunk(chunk_text, query="What is backpropagation?")

    # Filter a list of chunks to the most relevant
    kept = filter_chunks(chunks, query="neural network training")

    # Probe what the model already knows
    knowledge = probe("What is the Eiffel Tower?")

The MRSF delta-compression storage system (mrsf_write, mrsf_read, etc.)
is available under pymrsf.experimental as a research-grade backend.
"""

import logging
import os
from dataclasses import dataclass, field

from . import experimental

from . import cache
from .cache import (
    clear_cache,
    clear_embedding_cache,
    configure_cache,
    get_cache_stats,
    get_embedding_cache_stats,
    reset_cache_stats,
)
from .chunker import smart_chunk
from .core import (
    ModelSession,
    compute_delta,
    detokenize,
    get_backend,
    get_model_version,
    get_provider,
    get_raw_lm,
    get_surprises,
    next_token_greedy,
    provider_capabilities,
    quantized_argmax,
    set_provider,
    tokenize,
)
from .embeddings import embed, get_embedding_dim
from .probe import probe, probe_compare
from .rag import (
    DEFAULT_RAG_THRESHOLDS,
    DEFAULT_RELEVANCE_CUTOFF,
    DEFAULT_WEIGHTS,
    WeightConfig,
    explain_chunk,
    filter_chunks,
    filter_chunks_async,
    score_chunk,
    score_chunk_async,
    score_chunks,
    score_chunks_async,
    score_chunks_batch,
    smart_filter,
)

__version__ = "1.0.1"


# ── Centralized runtime configuration ─────────────────────────────────────────

@dataclass
class Config:
    """Central configuration for all pymrsf settings.

    Reads defaults from environment variables so existing .env setups
    continue to work. Override at runtime with configure().

    All fields are live-reconfigurable — model state is invalidated
    automatically on changes to model-affecting fields.
    """
    provider: str = field(default_factory=lambda: os.getenv("PYMRSF_PROVIDER", "local").lower())
    model_path: str = field(
        default_factory=lambda: os.getenv("PYMRSF_MODEL_PATH", "./models/mistral-7b-v0.1.Q4_K_M.gguf")
    )
    n_ctx: int = field(default_factory=lambda: int(os.getenv("PYMRSF_N_CTX", "4096")))
    n_gpu_layers: int = field(default_factory=lambda: int(os.getenv("PYMRSF_N_GPU_LAYERS", "0")))
    n_threads: int = field(default_factory=lambda: int(os.getenv("PYMRSF_N_THREADS", str(os.cpu_count() or 4))))
    ollama_base: str = field(default_factory=lambda: os.getenv("PYMRSF_OLLAMA_BASE", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: os.getenv("PYMRSF_EMBED_MODEL", "nomic-embed-text"))
    embed_timeout: int = field(default_factory=lambda: int(os.getenv("PYMRSF_EMBED_TIMEOUT", "30")))
    embed_dim: int = field(default_factory=lambda: int(os.getenv("PYMRSF_EMBED_DIM", "768")))
    model_version: str = field(default_factory=lambda: os.getenv("PYMRSF_MODEL_VERSION", ""))
    surprise_threshold: float = field(default_factory=lambda: float(os.getenv("PYMRSF_SURPRISE_THRESHOLD", "-1.0")))
    allow_provider_fallback: bool = field(
        default_factory=lambda: os.getenv("PYMRSF_ALLOW_PROVIDER_FALLBACK", "false").lower() == "true"
    )
    logit_precision: int = field(default_factory=lambda: int(os.getenv("PYMRSF_LOGIT_PRECISION", "6")))
    db_path: str = field(default_factory=lambda: os.getenv("PYMRSF_DB_PATH", "mrsf.db"))
    faiss_path: str = field(default_factory=lambda: os.getenv("PYMRSF_FAISS_PATH", "mrsf.faiss"))
    default_relevance_cutoff: float = 0.30
    default_min_rag_score: int = 50
    default_diversity_threshold: float = 0.85


_config = Config()
_MODEL_AFFECTING_FIELDS = frozenset({"provider", "model_path", "n_ctx", "n_gpu_layers", "n_threads"})


def configure_logging(level: str = "INFO") -> None:
    """Configure the pymrsf logger.

    Call this once at application startup to see library log output.
    By default pymrsf ships with a NullHandler so it never pollutes stdout
    unless the application explicitly enables logging.

    Args:
        level: Log level string — "DEBUG", "INFO", "WARNING", "ERROR".

    Example:
        import pymrsf
        pymrsf.configure_logging("DEBUG")   # show all internal messages
        pymrsf.configure_logging("WARNING") # only warnings and errors
    """
    pkg_logger = logging.getLogger("pymrsf")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def configure(**kwargs) -> Config:
    """Set global pymrsf configuration at runtime.

    Accepts any field name from Config. When a model-affecting field changes
    (provider, model_path, n_ctx, n_gpu_layers, n_threads), invalidates the
    cached model state so the next call re-loads with the new settings.

    Example:
        import pymrsf
        pymrsf.configure(provider="openai", default_relevance_cutoff=0.40)
    """
    global _config
    model_affected = False
    provider_changed = False

    for k, v in kwargs.items():
        if not hasattr(_config, k):
            raise ValueError(f"Unknown config key: '{k}'. Valid keys: {list(_config.__dataclass_fields__)}")
        if k in _MODEL_AFFECTING_FIELDS:
            old = getattr(_config, k)
            if old != v:
                model_affected = True
                if k == "provider":
                    provider_changed = True
                    v = v.lower()
        setattr(_config, k, v)

    if model_affected:
        from . import core as _core
        _core._lm = None
        _core._lm_loaded = False
        _core._backend = None

    if provider_changed:
        os.environ["PYMRSF_PROVIDER"] = _config.provider

    return _config


def get_config() -> Config:
    """Return the current global Config instance."""
    return _config

# ── Public API ────────────────────────────────────────────────────────────────

def __getattr__(name):
    """Resolve LOGIT_PRECISION live via core (not captured at import time)."""
    if name == "LOGIT_PRECISION":
        from .core import _logit_precision
        return _logit_precision()
    raise AttributeError(f"module 'pymrsf' has no attribute {name!r}")


__all__ = [
    # ── Chunking (headline API) ──────────────────────────────────────────────────
    "smart_chunk",

    # ── RAG scoring ──────────────────────────────────────────────────────────────
    "score_chunk",
    "score_chunks",
    "score_chunks_batch",
    "explain_chunk",
    "filter_chunks",
    "smart_filter",
    "DEFAULT_WEIGHTS",
    "DEFAULT_RELEVANCE_CUTOFF",
    "DEFAULT_RAG_THRESHOLDS",
    "WeightConfig",

    # Async variants
    "score_chunk_async",
    "score_chunks_async",
    "filter_chunks_async",

    # Knowledge probing
    "probe",
    "probe_compare",

    # Embeddings
    "embed",
    "get_embedding_dim",

    # ── Core / backend ─────────────────────────────────────────────────────────
    "tokenize",
    "detokenize",
    "quantized_argmax",
    "get_surprises",
    "compute_delta",
    "next_token_greedy",
    "ModelSession",
    "get_backend",
    "get_raw_lm",
    "provider_capabilities",
    "set_provider",
    "get_provider",
    "get_model_version",
    "LOGIT_PRECISION",

    # ── Cache ──────────────────────────────────────────────────────────────────
    "cache",
    "configure_cache",
    "get_cache_stats",
    "get_embedding_cache_stats",
    "reset_cache_stats",
    "clear_cache",
    "clear_embedding_cache",

    # ── Runtime configuration ──────────────────────────────────────────────────
    "Config",
    "configure",
    "get_config",
    "configure_logging",

    # ── Experimental research backend ──────────────────────────────────────────
    "experimental",

    # Version
    "__version__",
]
