# pymrsf — Novelty-Aware RAG Chunk Scoring

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)]()

**Stop wasting context window on information your LLM already knows.**

`pymrsf` scores RAG chunks by measuring **information gain** — not just relevance. It uses the model's own predictive surprise to detect which chunks contain genuinely new information.

**🚀 New in v0.4:** Lightweight API providers (OpenAI, Anthropic) — get started in 30 seconds without downloading a 4GB model!

## The Problem

Standard RAG retrieves chunks by *relevance* (cosine similarity). But a chunk can be highly relevant while containing *only facts the model already memorized during training*. You waste precious context window on redundant information.

Also, if the LLM already *knows the answer* to the query, even novel chunks are less useful. And if two chunks say the same thing, you don't need both.

## The Solution

`pymrsf` introduces **multi-factor novelty-aware scoring**:

| Factor | What It Measures | Weight |
|--------|-----------------|--------|
| **Novelty** | How much *new* information does this chunk contain? | 40% |
| **Relevance** | How related is this chunk to the query? | 40% |
| **Query Ignorance** | Does the model *not* know the answer to your question? | 20% |
| **Diversity** | Does a better chunk already cover this content? | Dedup |

## Quick Start

### 🚀 Get Started in 30 Seconds (No 4GB Model Download!)

The easiest way to try `pymrsf` is with an API provider:

```bash
# Install with OpenAI support (no heavy dependencies!)
pip install pymrsf[openai]

# Set your API key
export OPENAI_API_KEY='sk-...'
```

```python
import os
os.environ["PYMRSF_PROVIDER"] = "openai"  # Use OpenAI instead of local model

from pymrsf.rag import filter_chunks

chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
]

query = "How does backpropagation work?"
useful = filter_chunks(chunks, query, min_rag_score=50, verbose=True)
# → Returns only relevant chunks, saves your context window!
```

### 🏠 Local Model (Advanced Features)

For full features including **knowledge probing** and **delta compression**, use a local model:

```bash
# Install with local model support
pip install pymrsf[local]

# Download a model (one-time, ~4GB)
# Example: Mistral 7B Q4 from https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF
```

```python
import os
os.environ["PYMRSF_PROVIDER"] = "local"  # default
os.environ["PYMRSF_MODEL_PATH"] = "./models/mistral-7b-v0.1.Q4_K_M.gguf"

from pymrsf.rag import filter_chunks
from pymrsf import probe

# RAG scoring (same as API mode)
useful = filter_chunks(chunks, query, min_rag_score=50)

# Knowledge probing (local only)
result = probe("To be or not to be, that is the question.")
print(f"Knowledge: {result['knowledge_score']}/100")  # 92/100 (memorized)
```

## Installation Options

Choose based on your needs:

```bash
# Lightweight — OpenAI API (recommended for getting started)
pip install pymrsf[openai]

# Lightweight — Anthropic API  
pip install pymrsf[anthropic]

# Full features — Local model (4GB+ model download required)
pip install pymrsf[local]

# Everything — All providers
pip install pymrsf[all]

# Development
git clone https://github.com/riiseup08/mrsf.git
cd mrsf
pip install -e .[all]
```

### Provider Comparison

| Feature | Local | OpenAI | Anthropic |
|---------|-------|--------|-----------|
| RAG Chunk Scoring | ✅ | ✅ | ✅ |
| Knowledge Probing | ✅ | ❌ | ❌ |
| Delta Compression | ✅ | ❌ | ❌ |
| Async Support | ✅ | ✅ | ✅ |
| Caching | ✅ | ✅ | ✅ |
| Setup Difficulty | Hard | Easy | Easy |
| Cost | Free | $$ | $$ |
| Privacy | Private | API | API |

## Features

### 🎯 RAG Chunk Scoring (Core Feature)

```python
from pymrsf.rag import score_chunk, score_chunks, score_chunks_batch

# Single chunk scoring
result = score_chunk(
    "Backpropagation computes gradients using the chain rule.",
    query="How does backpropagation work?",
    verbose=True
)
print(result["rag_score"])    # 72/100
print(result["verdict"])      # "good"
print(result["query_knowledge"])  # how much model knows the query

# Batch scoring (3-5x faster for many chunks)
results = score_chunks_batch(chunks, query)

# Custom weights (adjust the formula)
weights = {"novelty": 0.5, "relevance": 0.3, "query_ignorance": 0.2}
result = score_chunk(chunk, query, weights=weights)
```

### 🔍 Knowledge Probing

```python
from pymrsf import probe

result = probe("To be or not to be, that is the question.")
print(f"Knowledge: {result['knowledge_score']}/100 ({result['label']})")
# → Knowledge: 92/100 (memorized) — Shakespeare is well-known

result = probe("My proprietary algorithm uses a novel attention mechanism.")
print(f"Knowledge: {result['knowledge_score']}/100 ({result['label']})")
# → Knowledge: 15/100 (unknown) — novel content!
```

### 🔧 RAG Pipeline Filter

```python
from pymrsf.rag import filter_chunks

chunks = retriever.get(query, top_k=20)   # your retriever

# Only keep chunks worth sending to the LLM
good = filter_chunks(
    chunks,
    query,
    min_rag_score=50,      # skip low-value chunks
    top_k=5,                # limit context window usage
    diversity_threshold=0.85,  # dedup similar chunks
    verbose=True,
)

answer = llm.complete(query, context=good)
```

### ⚡ Performance Features (NEW in v0.4)

**Async Support** — Non-blocking scoring for production RAG pipelines:

```python
import asyncio
from pymrsf.rag import score_chunk_async, filter_chunks_async

async def my_rag_pipeline(chunks, query):
    # Score chunks without blocking
    useful = await filter_chunks_async(
        chunks,
        query,
        min_rag_score=50,
        max_concurrent=10,  # Score 10 chunks at once
    )
    return useful

# Run it
useful = asyncio.run(my_rag_pipeline(chunks, query))
```

**Caching** — Avoid re-scoring the same chunks:

```python
from pymrsf import cache

# Configure cache (do this once at startup)
cache.configure_cache(
    enabled=True,
    max_size=10000,  # Store up to 10k scored chunks
    ttl=3600,        # Cache for 1 hour
)

# Score chunks (caching happens automatically)
result = score_chunk(chunk, query=query)

# Check cache performance
cache.print_cache_stats()
# → Hit rate: 85.3% (cache is working!)

# Clear cache if needed
cache.clear_cache()
```

**Performance Benefits:**
- 🚀 **Async**: 3-5x faster with API providers (OpenAI, Anthropic)
- 💾 **Caching**: Near-instant for repeated chunks (~100x speedup)
- 🔄 **Combined**: Up to 10x throughput improvement in production

**Note**: Async is designed for API providers. Local models work best with synchronous functions.

See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks and best practices.

See [example_performance.py](example_performance.py) for runnable benchmarks.

### 📦 Delta Compression (Experimental)

Store text efficiently using LLM surprises:

```python
from pymrsf import mrsf_write, mrsf_read, save_index

# Write (stores only surprise tokens = ~40% compression)
mrsf_write("The Eiffel Tower is in Paris.")
save_index()

# Read (reconstructs from delta + model)
results = mrsf_read("famous landmark in France")
```

## Examples

See [example_openai.py](example_openai.py) for a complete example using the OpenAI provider.

For local model usage, see the examples in the Features section below.

## Configuration

### Using API Providers (Recommended for Getting Started)

Create a `.env` file (or copy from `.env.example`):

**OpenAI:**
```bash
# .env file
PYMRSF_PROVIDER=openai
OPENAI_API_KEY=sk-...
PYMRSF_MODEL_VERSION=gpt-3.5-turbo  # or gpt-4, gpt-4o
```

**Anthropic:**
```bash
# .env file
PYMRSF_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
PYMRSF_MODEL_VERSION=claude-3-5-sonnet-20241022  # or other Claude models
```

### Using Local Models (Advanced)

```bash
# .env file
PYMRSF_PROVIDER=local  # default
PYMRSF_MODEL_PATH=./models/mistral-7b-v0.1.Q4_K_M.gguf
PYMRSF_N_GPU_LAYERS=0  # set to 20-30 if you have GPU
PYMRSF_N_CTX=4096  # context window size
```

**Where to get local models:**
- [Mistral 7B GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF) (recommended)
- [Llama 2 7B GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
- Any GGUF model from [TheBloke on Hugging Face](https://huggingface.co/TheBloke)

## Scoring Concepts

### RAG Score Formula
```
rag_score = novelty × 0.40 + relevance × 0.40 + query_ignorance × 0.20
```

### What the Scores Mean

| Score | Verdict | Action |
|-------|---------|--------|
| 80-100 | Excellent | Prioritize this chunk |
| 60-79 | Good | Include in context |
| 40-59 | Moderate | Include if space allows |
| 20-39 | Weak | Skip if better chunks exist |
| 0-19 | Skip | Model already knows this |

## Project Structure

```
pymrsf/
├── __init__.py     # Public API exports
├── core.py         # Multi-provider backend (local, OpenAI, Anthropic) with lazy loading
├── embeddings.py   # Ollama embedding API client
├── probe.py        # Knowledge probing (local provider only)
├── rag.py          # RAG chunk scoring with novelty + relevance + diversity
├── storage.py      # Delta compression storage (local provider only, experimental)
├── inspect.py      # Token-level visualization tools
└── benchmark.py    # Compression/latency benchmarks
```

## Upgrading from v0.3

If you're upgrading from an earlier version, here's what changed in v0.4:

**Dependencies are now optional!**
```bash
# Old installation (still works, but heavy)
pip install -e .

# New installation (choose what you need)
pip install -e .[local]    # for local models
pip install -e .[openai]   # for OpenAI API
pip install -e .[all]      # everything
```

**Your existing code still works:**
- Default provider is still `local` (no breaking changes)
- If you have `llama-cpp-python` installed, everything works as before
- No code changes needed unless you want to use API providers

**To use API providers:**
```python
import os
os.environ["PYMRSF_PROVIDER"] = "openai"  # or "anthropic"
# Rest of your code stays the same!
```

## Project Status

**Alpha** — The RAG novelty scoring works and solves a real problem. The delta compression/storage system is experimental.

## License

MIT
