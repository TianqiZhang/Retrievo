# BEIR Benchmarks

Evaluation of HybridSearch.NET retrieval quality against [BEIR (Benchmarking Information Retrieval)](https://github.com/beir-cellar/beir) datasets.

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| `nfcorpus` (default) | Biomedical IR — 3,633 PubMed articles, 323 test queries, graded relevance (0/1/2) |
| `scifact` | Scientific claim verification — 5,183 abstracts, 300 test queries, binary relevance |
| `fiqa` | Financial opinion QA — 57,638 documents, 648 test queries |
| `trec-covid` | COVID-19 biomedical literature — 171,332 documents, 50 test queries |
| `hotpotqa` | Multi-hop QA — 5,233,329 documents, 7,405 test queries |
| `arguana` | Argument retrieval — 8,674 documents, 1,406 test queries |
| `quora` | Duplicate question retrieval — 522,931 documents, 10,000 test queries |

Any BEIR dataset ID can be used — the tool downloads from the BEIR CDN automatically.

## Running Benchmarks

```bash
# Run NFCorpus benchmark (default, auto-downloads on first run)
dotnet run --project benchmarks/HybridSearch.Benchmarks

# Run SciFact benchmark
dotnet run --project benchmarks/HybridSearch.Benchmarks -- --dataset scifact

# Run with precomputed embeddings (enables vector-only and hybrid modes)
dotnet run --project benchmarks/HybridSearch.Benchmarks -- --dataset scifact --embeddings scifact-embeddings.bin

# List all available datasets
dotnet run --project benchmarks/HybridSearch.Benchmarks -- --list-datasets
```

### CLI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--dataset` | `-d` | `nfcorpus` | BEIR dataset to evaluate |
| `--data-dir` | | `./benchmarks/data` | Base directory for dataset storage |
| `--embeddings` | | (none) | Path to pre-computed embeddings binary cache |
| `--list-datasets` | | | Show available datasets and exit |
| `--help` | `-h` | | Show usage and exit |

## Generating Embeddings

Without embeddings, the benchmark runs **lexical-only** (BM25). To enable vector-only and hybrid modes, pre-compute embeddings with the included Python script:

```bash
pip install openai azure-identity

# Generate embeddings for NFCorpus
python tools/generate_embeddings.py --data-dir benchmarks/data/nfcorpus --output nfcorpus-embeddings.bin

# Generate embeddings for SciFact
python tools/generate_embeddings.py --data-dir benchmarks/data/scifact --output scifact-embeddings.bin
```

Set `HYBRIDSEARCH_AZURE_OPENAI_ENDPOINT` to your Azure OpenAI endpoint. Authentication uses `DefaultAzureCredential` (Azure CLI, managed identity, etc.). Optionally set `HYBRIDSEARCH_AZURE_OPENAI_DEPLOYMENT` (defaults to `text-embedding-3-small`).

The script encodes all documents and queries using the specified model and saves them in the binary cache format. Then re-run the benchmark with `--embeddings <path>`.

## Results

### NFCorpus

| Configuration | nDCG@10 | MAP@10 | Recall@100 | Avg Query |
|---------------|---------|--------|------------|-----------|
| Lexical-only (BM25) | **0.325** | 0.239 | 0.250 | 2.4ms |
| Vector-only (text-embedding-3-small) | 0.384 | 0.291 | 0.360 | 7.5ms |
| Hybrid (BM25 + Vector) | 0.374 | 0.278 | 0.358 | 7.7ms |
| Hybrid (L=0.1, V=1.0) | **0.391** | 0.294 | 0.360 | 8.2ms |
| BEIR BM25 baseline (Anserini) | 0.325 | — | — | — |

### Analysis

Our BM25 implementation now **matches the reference Anserini/Lucene baseline** (0.325 nDCG@10), up from 0.304 in the initial implementation. This was achieved by:

1. **Tuning BM25 parameters** — k1=0.9, b=0.4 (matching Anserini defaults for BEIR)
2. **English stemming** — Porter stemmer with English possessive filter and stop words
3. **Bag-of-words query construction** — analyzer-based term extraction instead of query parser escaping

Vector-only retrieval with `text-embedding-3-small` (1536 dims) achieves **0.384 nDCG@10**, outperforming published baselines including ColBERT-v2 (0.338) on this dataset. Hybrid search with tuned weights (lexical=0.1, vector=1.0) pushes this further to **0.391 nDCG@10**.

## How It Works

The benchmark parses standard BEIR files (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`), builds a HybridSearch index from the corpus, runs all test queries, and computes:

- **nDCG@10** — normalized discounted cumulative gain at rank 10
- **MAP@10** — mean average precision at rank 10
- **Recall@100** — fraction of relevant documents retrieved in top 100

Each dataset is evaluated in up to three modes: lexical-only (BM25), vector-only (cosine similarity), and hybrid (RRF fusion). Vector and hybrid modes require pre-computed embeddings.
