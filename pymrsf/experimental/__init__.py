"""
pymrsf.experimental — Research-grade MRSF storage backend.

This subpackage contains the delta-compression storage system (MRSF),
inspection utilities, and benchmarking tools. These are research-grade
and may change between minor versions.

For production use, prefer the RAG scoring and chunking APIs in the
top-level pymrsf namespace.
"""

from .storage import (
    mrsf_write,
    mrsf_read,
    mrsf_read_novel,
    mrsf_delete,
    rebuild_index,
    save_index,
    load_index,
    reset_index_metadata,
    close_connections,
)
from .inspect import mrsf_inspect, mrsf_rebuild_explained
from .benchmark import mrsf_benchmark_canterbury, mrsf_latency_benchmark

__all__ = [
    "mrsf_write",
    "mrsf_read",
    "mrsf_read_novel",
    "mrsf_delete",
    "rebuild_index",
    "save_index",
    "load_index",
    "reset_index_metadata",
    "close_connections",
    "mrsf_inspect",
    "mrsf_rebuild_explained",
    "mrsf_benchmark_canterbury",
    "mrsf_latency_benchmark",
]
