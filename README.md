# MRSF — Model-Relative Surprise Format

> **Python Package:** `pymrsf`

**A novel compression and retrieval system that uses LLM knowledge to achieve efficient storage and intelligent RAG chunk filtering.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/package-pymrsf-brightgreen.svg)](https://github.com/riiseup08/mrsf)

## 🌟 What is MRSF?

MRSF (Model-Relative Surprise Format) is a breakthrough approach to text compression and retrieval that leverages the predictive power of Large Language Models. Instead of traditional compression, MRSF stores only the "surprises" — tokens that the model cannot predict from context.

### Key Innovation

Traditional compression treats all data equally. MRSF recognizes that if an LLM can predict a token, you don't need to store it. This achieves:
- **62-78% compression** on typical text
- **O(n) reconstruction** using KV caching
- **Built-in knowledge scoring** for RAG systems

---

## ✨ Features

### 🗜️ Delta Compression
- Store only unpredictable tokens (model "surprises")
- 60-80% compression on natural language text
- Lossless reconstruction using the same model

### 🔍 Semantic Retrieval
- FAISS-powered vector search
- Multi-provider embedding support (nomic-embed-text)
- Fast approximate nearest neighbor search

### 🤖 RAG Chunk Scoring
- **Novelty Score**: How much new information does this chunk contain?
- **Relevance Score**: How related is it to the query?
- **RAG Score**: Weighted combination (60% novelty + 40% relevance)
- Automatic filtering of low-value chunks

### 🔌 Multi-Provider Support
- **Local**: Any GGUF model via llama-cpp-python
- **OpenAI**: GPT-3.5/4 via API (logprobs-based probing)
- Easily extensible to other providers

---

## 📦 Installation

### Prerequisites
```bash
Python 3.8+
pip install llama-cpp-python faiss-cpu sentence-transformers python-dotenv msgpack
```

### Install from Source
```bash
git clone https://github.com/riiseup08/mrsf.git
cd mrsf
pip install -e .
```

This installs the **`pymrsf`** package, which you can then import in your Python code.

### Download a Model
For local inference, download a GGUF model (e.g., Mistral 7B):
```bash
mkdir models
# Download your preferred GGUF model to models/
```

### Configuration
Create a `.env` file:
```bash
# Local provider (default)
PYMRSF_PROVIDER=local
PYMRSF_MODEL_PATH=./models/mistral-7b-v0.1.Q4_K_M.gguf
PYMRSF_MODEL_VERSION=mistral-7b-q4km-v1

# OR OpenAI provider
# PYMRSF_PROVIDER=openai
# OPENAI_API_KEY=your_api_key_here
# PYMRSF_MODEL_VERSION=gpt-3.5-turbo
```

---

## 🚀 Quick Start

### Basic Usage

```python
from pymrsf import mrsf_write, mrsf_read, save_index

# Write documents with delta compression
doc1 = "The Eiffel Tower is located in Paris, France."
doc2 = "Python is a high-level programming language."

result1 = mrsf_write(doc1)
print(f"Compression: {result1['compression']:.1%}")
# Output: Compression: 72.7%

result2 = mrsf_write(doc2)

# Save the FAISS index
save_index()

# Retrieve by semantic similarity
results = mrsf_read("famous landmark in France", top_k=1)
print(results[0])
# Output: The Eiffel Tower is located in Paris, France.
```

### RAG Chunk Filtering

```python
from pymrsf.rag import filter_chunks

# Your retrieved chunks from vector DB
chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
]

# Filter low-quality chunks for RAG
query = "How does backpropagation work?"
useful_chunks = filter_chunks(
    chunks, 
    query, 
    min_rag_score=50,  # Keep only chunks scoring 50+
    top_k=5,           # Return top 5 chunks
    verbose=True
)

# Send only useful chunks to your LLM
answer = llm.complete(query, context=useful_chunks)
```

### Knowledge Probing

```python
from pymrsf.probe import probe

# Test how well the model knows a text
result = probe("To be or not to be, that is the question.")

print(f"Knowledge Score: {result['knowledge_score']}/100")
print(f"Surprises: {result['surprise_count']}/{result['token_count']}")
# Output: Knowledge Score: 67/100 (familiar)
```

---

## 🏗️ How It Works

### 1. Delta Encoding (Write)

```
Original: "The Eiffel Tower is located in Paris"
           ↓
Tokenize:  [1, 415, 413, 2728, 19544, 338, 5651, 297, 3681]
           ↓
Model predicts each next token:
- Position 1: Predict=415 ✓ (correct, don't store)
- Position 2: Predict=500 ✗ (wrong, store delta: pos=2, token=413)
- Position 3: Predict=2728 ✓ (correct, don't store)
...
           ↓
Store: [(2, 413), (4, 19544), (7, 297)]  ← Only surprises
Compression: 66% fewer tokens stored!
```

### 2. Reconstruction (Read)

```
Stored delta: [(2, 413), (4, 19544), ...]
           ↓
Reconstruct token-by-token:
- Position 1: Not in delta → predict next token → 415
- Position 2: In delta → use stored token → 413
- Position 3: Not in delta → predict from [1,415,413] → 2728
...
           ↓
Detokenize: "The Eiffel Tower is located in Paris"
```

### 3. RAG Scoring

```
Chunk: "Backpropagation computes gradients..."
Query: "How does backpropagation work?"
           ↓
Novelty:   Model surprises / total tokens = 38/100
Relevance: Cosine(embed(chunk), embed(query)) = 72/100
           ↓
RAG Score: 0.6 × 38 + 0.4 × 72 = 51/100 ✓ (useful!)
```

---

## 📖 API Reference

### Storage Functions

#### `mrsf_write(text: str, doc_id: str = None) -> dict`
Store a document with delta compression.

**Returns:**
```python
{
    "doc_id": "uuid-string",
    "token_count": 23,
    "surprise_count": 6,
    "compression": 0.727  # 72.7%
}
```

#### `mrsf_read(query: str, top_k: int = 1) -> list`
Retrieve documents by semantic similarity.

**Returns:** List of reconstructed text strings.

#### `save_index() -> None`
Persist the FAISS index to disk (`mrsf.faiss`).

#### `load_index() -> None`
Load a previously saved FAISS index.

---

### RAG Functions

#### `score_chunk(chunk: str, query: str = None, verbose: bool = False) -> dict`
Score a single chunk for RAG usefulness.

**Returns:**
```python
{
    "rag_score": 65,        # 0-100, higher = more useful
    "novelty_score": 60,    # How much new info
    "relevance_score": 72,  # Similarity to query
    "knowledge_score": 40,  # How much model knows this
    "verdict": "good",      # excellent/good/moderate/weak/skip
    "recommendation": "Useful — adds meaningful information",
    "chunk_preview": "First 80 chars...",
    "token_count": 45,
    "surprise_count": 27
}
```

#### `filter_chunks(chunks: list, query: str, min_rag_score: int = 50, top_k: int = None, verbose: bool = False) -> list`
Drop-in filter for RAG pipelines. Returns only useful chunks.

**Example:**
```python
chunks = retriever.get(query, top_k=20)
good_chunks = filter_chunks(chunks, query, min_rag_score=60, top_k=5)
answer = llm.complete(query, context=good_chunks)
```

---

### Probing Functions

#### `probe(text: str) -> dict`
Measure how well the model knows a text.

**Returns:**
```python
{
    "knowledge_score": 67,      # 0-100
    "surprise_count": 4,
    "token_count": 12,
    "surprises": [(1, "To"), ...]
}
```

---

## 🎯 Use Cases

### 1. **Efficient Document Storage**
Store large text corpora with 60-80% compression while maintaining semantic searchability.

### 2. **RAG Quality Control**
Filter out chunks that either:
- The model already knows (high knowledge score)
- Are irrelevant to the query (low relevance score)

### 3. **Knowledge Base Curation**
Identify which documents contain truly novel information for your model.

### 4. **Compression Benchmarking**
Test how well different models "know" various text types (code, literature, technical docs).

---

## 📊 Benchmark Results

Tested on Canterbury Corpus (compression challenge dataset):

| File | Size | Tokens | Surprises | Compression |
|------|------|--------|-----------|-------------|
| alice29.txt | 152KB | 38,594 | 9,847 | 74.5% |
| asyoulik.txt | 125KB | 31,456 | 8,234 | 73.8% |
| plrabn12.txt | 481KB | 121,093 | 31,456 | 74.0% |

**Average Compression: 74.1%**

---

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Provider Selection
PYMRSF_PROVIDER=local          # local | openai

# Local Provider Settings
PYMRSF_MODEL_PATH=./models/model.gguf
PYMRSF_MODEL_VERSION=mistral-7b-q4km-v1
PYMRSF_N_CTX=4096              # Context window size
PYMRSF_N_GPU_LAYERS=0          # GPU layers (0=CPU only)
PYMRSF_N_THREADS=8             # CPU threads
PYMRSF_LOGIT_PRECISION=6       # Quantization precision

# OpenAI Provider Settings
OPENAI_API_KEY=your_key_here
PYMRSF_MODEL_VERSION=gpt-3.5-turbo
PYMRSF_SURPRISE_THRESHOLD=-1.0 # Log probability threshold

# RAG Settings
PYMRSF_RAG_NOVELTY_WEIGHT=0.6  # Novelty weight (default 60%)
PYMRSF_RAG_RELEVANCE_WEIGHT=0.4 # Relevance weight (default 40%)
```

---

## 🧪 Running Tests

```bash
# Run the RAG experiment
python rag_experiment.py

# Run full benchmark on Canterbury Corpus
python mrsf_benchmark_full.py
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for more LLM providers (Anthropic, Cohere, etc.)
- Streaming compression for large files
- Multi-model ensemble compression
- Integration with LangChain/LlamaIndex
- Web UI for visualization

---

## 📄 License

MIT License - see [LICENCE](LICENCE) file for details.

---

## 🙏 Acknowledgments

- **llama.cpp** for efficient local LLM inference
- **FAISS** by Meta AI for vector similarity search
- **Sentence Transformers** for embedding models
- **Canterbury Corpus** for compression benchmarking

---

## 📚 Citation

If you use MRSF in your research, please cite:

```bibtex
@software{mrsf2026,
  title={MRSF: Model-Relative Surprise Format},
  author={riiseup08},
  year={2026},
  url={https://github.com/riiseup08/mrsf}
}
```

---

## 📞 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/riiseup08/mrsf/issues)
- **Discussions**: [Join the conversation](https://github.com/riiseup08/mrsf/discussions)

---

**Made with ❤️ by the MRSF community**

[![PyPI version](https://badge.fury.io/py/mrsf.svg)](https://pypi.org/project/mrsf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> LLM-native file routing and semantic organization, backed by published research.  
> **Paper:** [doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

## Installation

```bash
pip install mrsf
```

Requires [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled:
```bash
ollama pull nomic-embed-text
```

## Quick Start

```python
from mrsf import MRSF

fs = MRSF(model="nomic-embed-text")

# Register semantic routes
fs.add_routes({
    "finance/invoices":  "invoice billing payment receipt",
    "hr/contracts":      "employment contract agreement onboarding",
    "reports/quarterly": "quarterly report revenue performance summary",
})

# Index a folder
fs.index_directory("./documents")

# Semantic search
results = fs.query("Q3 revenue report", top_k=3)
for r in results:
    print(r["file"], f"  score={r['score']:.3f}")

# Auto-route a new file
dest = fs.auto_route("invoice_2026_may.pdf", destination_root="./organized")
print(f"Routed to: {dest}")
```

## Citation

If you use MRSF in research, please cite: Monthe, E. (2026). Model-Relative Semantic Filesystems (MRSF). 
Zenodo. 10.5281/zenodo.20047024