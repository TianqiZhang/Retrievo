# Retrievo

***Hybrid search for .NET — BM25 + vectors + RRF fusion, zero infrastructure***

[![NuGet](https://img.shields.io/nuget/vpre/Retrievo?color=blue)](https://www.nuget.org/packages/Retrievo) [![Downloads](https://img.shields.io/nuget/dt/Retrievo)](https://www.nuget.org/packages/Retrievo) [![License](https://img.shields.io/github/license/TianqiZhang/Retrievo)](https://github.com/TianqiZhang/Retrievo/blob/master/LICENSE) [![.NET](https://img.shields.io/badge/.NET-8.0-512BD4?logo=dotnet)](https://dotnet.microsoft.com/)

Retrievo is an open-source, in-process, in-memory search library for .NET that combines BM25 lexical matching with vector similarity search. Results are merged via [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) into a single ranked list — no external servers, no databases, no infrastructure. Designed for corpora up to ~10k documents: local agent memory, small RAG pipelines, developer tools, and offline/edge scenarios.

---

## Quick Install

```shell
dotnet add package Retrievo --prerelease
```

For Azure OpenAI embeddings:

```shell
dotnet add package Retrievo.AzureOpenAI --prerelease
```

---

## Key Features

### Core Search
- **Hybrid Retrieval**: Combine BM25 and cosine similarity using RRF fusion.
- **Standalone Modes**: Use lexical-only or vector-only search when needed.
- **Explain Mode**: Detailed score breakdown for every search result.
- **Fielded Search**: Title and body fields with independent boost weights.
- **Metadata Filters**: Exact-match key-value filtering post-fusion.

### Index Management
- **Fluent Builder**: Clean API for batch construction and folder ingestion.
- **Mutable Index**: Incremental upserts and deletes with thread-safe commits.
- **Zero Infrastructure**: Runs entirely in-process with no external dependencies.
- **Auto-Embedding**: Transparently embed documents at index time.

### Developer Experience
- **SIMD Accelerated**: Hardware-intrinsics for fast brute-force vector math.
- **Query Diagnostics**: Detailed timing breakdown for every pipeline stage.
- **Pluggable Providers**: Easy integration with any embedding model or API.
- **CLI Tool**: Powerful terminal interface for indexing and querying.

---

## Quick Start

Build an index and search in a few lines:

```csharp
using Retrievo;
using Retrievo.Models;

var index = new HybridSearchIndexBuilder()
    .AddDocument(new Document { Id = "1", Body = "Neural networks learn complex patterns." })
    .AddDocument(new Document { Id = "2", Body = "Kubernetes orchestrates container deployments." })
    .Build();

using var _ = index;
var response = index.Search(new HybridQuery { Text = "neural network training", TopK = 5 });

foreach (var r in response.Results)
    Console.WriteLine($"  {r.Id}: {r.Score:F4}");
```

---

## Azure OpenAI Embeddings

Plug in an embedding provider and Retrievo handles the rest — documents are embedded at build time, queries at search time.

```csharp
using Retrievo.AzureOpenAI;

var provider = new AzureOpenAIEmbeddingProvider(
    new Uri("https://your-resource.openai.azure.com/"),
    "your-api-key",
    "text-embedding-3-small");

// Documents are auto-embedded during build
using var index = await new HybridSearchIndexBuilder()
    .AddFolder("./docs")  // loads *.md and *.txt recursively
    .WithEmbeddingProvider(provider)
    .BuildAsync();

// Query text is automatically converted to a vector
var response = await index.SearchAsync(new HybridQuery { Text = "how to deploy", TopK = 5 });
```

---

## Architecture

```
HybridQuery
    |
    v
+-------------------+
|  HybridSearchIndex |  (orchestrator)
+-------------------+
    |           |
    v           v
+--------+  +----------+
| Lucene |  | Brute-   |
| BM25   |  | Force    |
| Search |  | Cosine   |
+--------+  +----------+
    |           |
    v           v
  ranked      ranked
  list        list
    \         /
     v       v
  +----------+
  | RRF      |
  | Fusion   |
  +----------+
       |
       v
  SearchResponse
```

[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) merges multiple ranked lists without score normalization: `score(doc) = Σ weight / (k + rank)`. Documents that rank high on *both* lexical and vector lists get the biggest boost — surfacing results that are semantically relevant *and* contain the right keywords.

---

## Benchmarks

### Retrieval Quality (NDCG@10)

Validated against [BEIR](https://github.com/beir-cellar/beir) with 245-configuration parameter sweeps per dataset:

| Dataset | BM25 | Vector-only | **Hybrid (default)** | Hybrid (tuned) | Anserini BM25 |
|---------|------|-------------|----------------------|----------------|---------------|
| NFCorpus | 0.325 | 0.384 | **0.392** | 0.392 | 0.325 |
| SciFact | 0.665 | 0.731 | **0.756** | 0.757 | 0.679 |

Default parameters (`LexicalWeight=0.5, VectorWeight=1.0, RrfK=20, TitleBoost=0.5`) tuned via cross-dataset harmonic mean optimization.

### Query Latency

3,000 documents × 768-dimensional embeddings (`text-embedding-3-small`):

| Operation | Latency |
|-----------|---------|
| Vector-only query | < 5 ms |
| Lexical-only query | < 5 ms |
| Hybrid query (BM25 + vector + RRF) | < 10 ms |
| Index build (3k docs) | < 2 s |

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | Done | MVP hybrid retrieval, CLI, Azure OpenAI provider |
| **Phase 2** | Done | Mutable index, fielded search, filters, diagnostics |
| **Phase 3** | Planned | Snapshot export and import |
| **Phase 4** | Planned | ANN support for larger corpora |

---

## Build & Test

Requires [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later.

```shell
dotnet build
dotnet test
```

194 tests covering retrieval, vector math, fusion, mutable index, and CLI integration — 0 warnings.

---

## License

[MIT](https://github.com/TianqiZhang/Retrievo/blob/master/LICENSE)
