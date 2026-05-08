# Changelog

All notable changes to pymrsf will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-05-07

### Added
- **OpenAI API support** — Use GPT-3.5/GPT-4 without downloading a local model
- **Anthropic API support** — Use Claude models for basic RAG scoring
- **Async support** — Non-blocking `score_chunk_async()`, `score_chunks_async()`, and `filter_chunks_async()`
  - Concurrent scoring with configurable `max_concurrent` parameter
  - 3-5x faster with API providers
  - Ideal for production RAG pipelines
- **Caching system** — Automatic caching of chunk scores and embeddings
  - Configurable LRU cache with TTL support
  - Thread-safe implementation
  - Near-instant scoring for repeated chunks (~100x speedup)
  - Cache statistics and monitoring via `cache.print_cache_stats()`
- Optional dependencies system — Install only what you need:
  - `pymrsf[local]` — Local model support (llama-cpp-python)
  - `pymrsf[openai]` — OpenAI API support
  - `pymrsf[anthropic]` — Anthropic API support
  - `pymrsf[all]` — Everything
- `.env.example` configuration file with examples for all providers
- `example_openai.py` — Complete example using OpenAI provider
- `example_performance.py` — Async and caching benchmarks
- Provider comparison table in README
- Migration guide for upgrading from v0.3

### Changed
- **Breaking (optional dependencies):** `llama-cpp-python` is now optional, not required
  - Existing users: Run `pip install pymrsf[local]` to restore full functionality
  - Default behavior unchanged if llama-cpp-python is already installed
- `score_chunk()` now includes `use_cache` parameter and returns `cached` field in results
- Improved error messages throughout the codebase
- Updated README with API-based quick start (30 seconds to get started!)
- Core module docstring updated with provider information
- Embedding calls now use caching automatically

### Fixed
- Better handling of missing dependencies with helpful error messages
- Clear distinction between features available in each provider

### Performance Improvements
- **3-5x faster** with async scoring on API providers
- **~100x faster** for cached chunks (repeated queries)
- **Up to 10x throughput** improvement with async + caching combined
- Automatic embedding caching reduces redundant API calls

### Limitations by Provider
- **Local:** Full feature support (RAG scoring, knowledge probing, delta compression)
- **OpenAI:** Basic RAG scoring with logprobs API (no full knowledge probing)
- **Anthropic:** Relevance-only RAG scoring (no logprobs available from API)

## [0.3.0] and earlier

See git history for changes prior to v0.4.0.
