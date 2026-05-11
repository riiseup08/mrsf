# pymrsf — Model-Relative Semantic Filtering

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/LICENSE)
[![CI](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/actions/workflows/ci.yml/badge.svg)](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-95%20passing-brightgreen)]()

**Split text at knowledge boundaries — then score by information gain.**

Semantic chunkers split at sentence or token boundaries. `smart_chunk` uses the model's own predictive surprise to detect where one topic ends and another begins — producing chunks that align with actual knowledge transitions, not arbitrary token counts.

- **Surprise-guided chunking** — finds natural topic boundaries using the model's prediction signal
- **Novelty scoring** — measures whether the model already knows a chunk (surprise-based)
- **Relevance scoring** — cosine similarity against your query
- **Query Ignorance** — does the model even know the answer? (probe-based gate)
- **Diversity** — dedup post-filter to avoid redundant chunks

---

## Quick install

```bash
# Start fast with an API provider (no 4 GB model download)
pip install pymrsf[openai]
export OPENAI_API_KEY='sk-...'

# Or for full features (probing, smart_chunk, round-trip):
pip install pymrsf[local]
```

All providers require [Ollama](https://ollama.ai/) for embeddings:

```bash
ollama pull nomic-embed-text
```

---

## 30-second example — surprise-guided chunking

Instead of splitting at fixed sizes or sentence boundaries, `smart_chunk` uses the model's surprise signal to find natural knowledge transitions:

```python
from pymrsf import smart_chunk

long_article = """
Quantum computing leverages superposition and entanglement to perform
calculations that would be infeasible for classical computers. Unlike
classical bits, qubits can exist in multiple states simultaneously.
...
Machine learning models learn patterns from data through iterative
optimization of a loss function. Neural networks, in particular,
use backpropagation to adjust millions of parameters.
...
"""

# Chunks split at the boundary between "quantum computing" and "ML" —
# where the model's surprise signal drops after absorbing one topic
pieces = smart_chunk(long_article, min_chunk_len=200, max_chunk_len=800)
```

Works with any provider — uses surprise signals (local) or embedding similarity (API) to detect topic boundaries. Falls back to sentence splitting if neither is available.

---

## 60-second example — score and filter chunks

```python
from pymrsf import score_chunk, filter_chunks

chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
]

# Score a single chunk
result = score_chunk(chunks[0], query="How does backpropagation work?")
print(result["rag_score"])   # 0–100
print(result["verdict"])     # "excellent" / "good" / "moderate" / "weak" / "skip"

# Filter to only the useful chunks
useful = filter_chunks(chunks, query="How does backpropagation work?", min_rag_score=50)
# useful ≈ ["Backpropagation computes gradients..."]
```

With **async** for production pipelines:

```python
import asyncio
from pymrsf import filter_chunks_async

useful = asyncio.run(filter_chunks_async(chunks, query="...", min_rag_score=50))
```

---

## Provider matrix

| Feature | local | openai | anthropic |
|---------|-------|--------|-----------|
| **smart_chunk** (topic-aware) | Surprise-guided | Embedding similarity | Embedding similarity |
| **RAG scoring** | Full (novelty + relevance + ignorance) | Relevance-only | Relevance-only |
| **Knowledge probing** | ✅ Full | ⚠️ Limited | ❌ |
| **Delta compression / round-trip** | Experimental | ❌ | ❌ |
| **Model session** (KV-cache) | ✅ Yes | ❌ | ❌ |
| **Async scoring** | ✅ | ✅ | ✅ |
| **Score caching** | ✅ | ✅ | ✅ |

**Key takeaway:** Probing and surprise-guided chunking require the **local** provider (`pip install pymrsf[local]` + a GGUF model). Relevance scoring and embedding-based topic chunking work with any provider.

---

## Production configuration

```python
import pymrsf

# Enable pymrsf log output (silent by default)
pymrsf.configure_logging("INFO")

# Tweak runtime settings without touching env vars
pymrsf.configure(
    provider="openai",
    embed_timeout=60,
    default_relevance_cutoff=0.4,
)
```

Environment variables for container/CI environments:

```bash
PYMRSF_PROVIDER=openai
OPENAI_API_KEY=sk-...
PYMRSF_ALLOW_PROVIDER_FALLBACK=true   # silently fall back on embed failures
PYMRSF_EMBED_TIMEOUT=30
```

- `PYMRSF_ALLOW_PROVIDER_FALLBACK` — when `true`, embed failures log a warning and continue instead of raising. Off by default (fail-fast).
- `pymrsf.configure_logging("WARNING")` — pymrsf ships with a `NullHandler` so `import pymrsf` is silent until you opt in.

See [ENV_CONFIG.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/ENV_CONFIG.md) for all supported variables.

---

## Experimental: MRSF delta-compression storage

A research backend that stores only "surprise" tokens (40–60% compression) and reconstructs text via model inference. Import from `pymrsf.experimental`:

```python
from pymrsf.experimental import mrsf_write, mrsf_read, save_index

doc = mrsf_write("The Eiffel Tower was built in 1889.")
print(doc["compression"])   # 0.47 — 47% of tokens were predictable

save_index()
results = mrsf_read("famous French landmark", top_k=1)
```

[Full experimental docs →](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/tree/main/pymrsf/experimental)

---

## Score interpretation

| Score | Verdict | Suggested action |
|-------|---------|-----------------|
| 80–100 | excellent | Prioritise |
| 60–79 | good | Include |
| 40–59 | moderate | Include if space allows |
| 20–39 | weak | Skip if better chunks exist |
| 0–19 | skip | Model already knows this |

---

## Additional documentation

- [PROVIDER_SUPPORT.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/PROVIDER_SUPPORT.md) — full capability matrix with programmatic checks
- [ENV_CONFIG.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/ENV_CONFIG.md) — all environment variables
- [docs/CONCURRENCY.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/docs/CONCURRENCY.md) — threading and process-safety model
- [CHANGELOG.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/CHANGELOG.md) — version history

## Paper

The technical approach is described in the MRSF paper (link forthcoming). For now, see [CHANGELOG.md](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/blob/main/CHANGELOG.md) for the research lineage and [the experimental module](https://github.com/riiseup08/Model-Relative-Semantic-Filesystems/tree/main/pymrsf/experimental) for the delta-compression implementation.

## License

MIT
