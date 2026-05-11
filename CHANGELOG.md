# Changelog

All notable changes to pymrsf will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] — 2026-05-10

### Fixed
- README test badge updated (95 passing) and Config section added.
- README changes appear on PyPI (re-uploaded with version bump).

---

## [1.0.0] — 2026-05-10

### Stable release — production-ready RAG chunk scoring

The `pymrsf` library is now stable. All core APIs are solidified, the config
system is live-reconfigurable, and provider switching works without stale
bindings or import-time capture bugs.

### Major changes since v0.7.0

- **Live-reconfigurable Config** — `pymrsf.configure()` now invalidates model
  state automatically when provider or model-path changes. Module-level constants
  (`PROVIDER`, `LOGIT_PRECISION`, `MODEL_VERSION`) are deprecated but continue
  to resolve via PEP 562 `__getattr__` shims.
- **Provider switching works** — `set_provider()` and `configure()` use
  identical reset logic. No more stale probe bindings or cache-key mismatches.
- **Embedding API provider routing** — Anthropic's deprecated embedding API
  is replaced with Ollama routing. OpenAI embedding guard prevents
  Ollama-shaped model names from being sent to OpenAI.
- **OpenAI `get_surprises` fixed** — Now uses the legacy completions endpoint
  (`echo=True`, `logprobs=5`) for true per-position logprobs. Chat models
  raise a clear `NotImplementedError` with remediation options.
- **`compute_conditional_novelty` flag** — Opt-in to avoid triple-probe cost
  (default `False`). Propagated through all sync/async functions.
- **`score_chunks_batch` simplified** — Now delegates to `score_chunks`.
- **Deadlock-free embeddings** — Benign race pattern replaces double-checked
  locking on `_embed_dim_cache`.
- **API skip-gate fixed** — Query-ignorance gate correctly skips only when
  probe data is available.

### New Config fields
- `model_version` — explicit model override (env `PYMRSF_MODEL_VERSION`)
- `n_threads` — CPU thread count for local model (env `PYMRSF_N_THREADS`)
- `surprise_threshold` — surprise logprob threshold (env `PYMRSF_SURPRISE_THRESHOLD`)
- `allow_provider_fallback` — graceful fallback on embed failure (env `PYMRSF_ALLOW_PROVIDER_FALLBACK`)

### Backward compat
- `from pymrsf.core import PROVIDER` / `LOGIT_PRECISION` / `MODEL_VERSION`
  still works (resolves live via `__getattr__`).
- Env-var-only setups continue to work without calling `configure()`.
- `set_provider("openai")` still works (delegates to `configure`).

### Tests
- 95 tests, 0 failures.
- Smoke tests covering configure → set_provider → backward-compat → get_surprises.

---

## [0.7.0] — 2026-05-10

### Breaking changes
- **Delta compression removed from top-level API.** `mrsf_write`, `mrsf_read`,
  `mrsf_delete`, `rebuild_index`, `save_index`, `load_index`, `close_connections`,
  and related functions are no longer re-exported from `pymrsf`. Import from
  `pymrsf.experimental` instead.
- `rebuild_faiss_from_sqlite()` deprecated alias removed.
- **Cross-model-version enforcement (Property P1).** Reading delta-compressed
  records under a different `MODEL_VERSION` now raises `ValueError` instead of
  logging a warning. Records are bound to the model version that created them.

### New features
- **`smart_chunk` works with all providers.** API providers (OpenAI, Anthropic)
  that lack token-level logprobs now use embedding cosine similarity between
  sliding windows to detect topic boundaries. Works wherever embeddings are
  available; only falls back to sentence splitting when neither logits nor
  embeddings work.
- **BEIR retrieval benchmark** (`benchmarks/beir_eval.py`). Downloads BEIR
  datasets, indexes with MRSF + FAISS-only baseline, computes nDCG@10/Recall@10.
  Includes automatic dataset download (fixed SSL via `requests`), nested zip
  extraction handling, and docs generation to `docs/benchmarks/retrieval.md`.

### Docs
- README reorganized: `smart_chunk` is now the headline feature. Provider matrix
  reordered. Delta compression section shortened and marked experimental.

### Tests
- 85 tests, 0 failures (up from 83).
- `test_cross_version_read_raises` — validates Property P1 (version mismatch → `ValueError`).

---

## [0.5.0] — 2026-05-10

### Breaking changes
- `pymrsf.storage`, `pymrsf.inspect`, `pymrsf.benchmark` moved to `pymrsf.experimental`.
  Top-level re-exports remain for backward compat and will be removed in v0.6.
- `rebuild_faiss_from_sqlite()` now emits `DeprecationWarning`. Use `reset_index_metadata()`.
- Default embed behavior is **fail-fast** on provider failure. Set
  `PYMRSF_ALLOW_PROVIDER_FALLBACK=true` to restore silent fallback (now logged as WARNING).

### New features
- `pymrsf.experimental` subpackage — MRSF storage clearly scoped as research-grade.
- `pymrsf.configure_logging(level)` — opt-in to library log output; import is now silent.
- `pymrsf.configure(embed_timeout=…, embed_model=…)` now takes effect at the next call.
- HTTP retry on transient embed failures: 3 attempts, exponential backoff (tenacity).
- `docs/CONCURRENCY.md` — threading model, WAL mode, multi-process patterns documented.

### Fixes
- Score cache: O(1) LRU eviction via `OrderedDict` (was O(n) min-timestamp scan).
- Score cache: `deepcopy` moved outside lock — callers no longer serialize on large objects.
- Score cache: removed unbounded `@lru_cache` on `_cached_text_hash`.
- `_embed_dim_cache` initialised under `threading.Lock` (double-checked locking).
- `smart_chunk` offset recovery: O(n) incremental walk (was O(n²) prefix re-detokenize).
- All `print()` calls converted to `logging` — `import pymrsf` is now side-effect free.

### Tests
- 83 tests, 0 failures (up from 66).
- Hypothesis round-trip property tests for MRSF storage (50 examples; 1000-example slow variant).
- Cache perf + concurrency tests: insertion <100 µs, 8-thread mixed get/set, deepcopy isolation.
- Config live-wiring tests, chunker O(n) perf test.

### Dependencies added
- `tenacity>=8.0.0` (core), `hypothesis>=6.0` (dev)
- `llama-cpp-python` tightened to `>=0.2.50,<0.3.0`

---

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
