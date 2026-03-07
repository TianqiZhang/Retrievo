# Retrievo

***Hybrid search for .NET — BM25 + vectors + RRF fusion, zero infrastructure***

[![NuGet Retrievo](https://img.shields.io/nuget/vpre/Retrievo?color=blue&label=Retrievo)](https://www.nuget.org/packages/Retrievo) [![NuGet Retrievo.AzureOpenAI](https://img.shields.io/nuget/vpre/Retrievo.AzureOpenAI?color=blue&label=Retrievo.AzureOpenAI)](https://www.nuget.org/packages/Retrievo.AzureOpenAI) [![Downloads](https://img.shields.io/nuget/dt/Retrievo)](https://www.nuget.org/packages/Retrievo) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/TianqiZhang/Retrievo/blob/master/LICENSE) [![.NET](https://img.shields.io/badge/.NET-8.0-512BD4?logo=dotnet)](https://dotnet.microsoft.com/)

Retrievo is an open-source, in-process, in-memory search library for .NET that combines BM25 lexical matching with vector similarity search. Results are merged via [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) into a single ranked list — no external servers, no databases, no infrastructure. Designed for corpora up to ~10k documents: local agent memory, small RAG pipelines, developer tools, and offline/edge scenarios.

---

## Quick Start

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

## Benchmarks

### Retrieval Quality (NDCG@10)

Validated against [BEIR](https://github.com/beir-cellar/beir) with 245-configuration parameter sweeps per dataset:

| Dataset | BM25 | Vector-only | **Hybrid (default)** | Hybrid (tuned) | Anserini BM25 |
|---------|------|-------------|----------------------|----------------|---------------|
| NFCorpus | 0.330 | 0.384 | **0.392** | 0.392 | 0.325 |
| SciFact | 0.685 | 0.731 | **0.756** | 0.757 | 0.679 |

Default parameters (`LexicalWeight=0.5, VectorWeight=1.0, RrfK=20, TitleBoost=0.5`) tuned via cross-dataset harmonic mean optimization.

### Query Latency

Measured on BEIR datasets with `text-embedding-3-small` (1536-dim) embeddings:

| Operation | NFCorpus (3.6k docs) | SciFact (5.2k docs) |
|-----------|---------------------|---------------------|
| Lexical-only (BM25) | 2.3 ms | 3.2 ms |
| Vector-only | 1.8 ms | 2.7 ms |
| Hybrid (BM25 + vector + RRF) | 2.2 ms | 3.3 ms |

Details: [`benchmarks/`](benchmarks/)
The benchmark harness can also export a temporary snapshot, re-import it, and assert identical ranked outputs per query with `--verify-snapshot-roundtrip`.
Micro-benchmark timings are hardware-dependent; the current vector math measurements live in [`benchmarks/README.md`](benchmarks/README.md).

---

## Key Features

### Core Search
- **Hybrid Retrieval**: Combine BM25 and cosine similarity using RRF fusion.
- **Standalone Modes**: Use lexical-only or vector-only search when needed.
- **Explain Mode**: Detailed score breakdown for every search result.
- **Fielded Search**: Title and body fields with independent boost weights.
- **Metadata Filters**: Exact-match, range, and contains filtering post-fusion.
- **Field Definitions**: Declare field types (`String`, `StringArray`) at index time for automatic filter semantics.
- **Finite Vector Validation**: Rejects NaN/Infinity embeddings and query vectors with clear exceptions.

### Index Management
- **Fluent Builder**: Clean API for batch construction and folder ingestion.
- **Mutable Index**: Incremental upserts and deletes with thread-safe commits.
- **Snapshot Export/Import**: Persist versioned JSON snapshots and reload indexes without rescanning source files.
- **Unique Document IDs**: Builders reject duplicate document IDs within a single build.
- **Zero Infrastructure**: Runs entirely in-process with no external dependencies.
- **Auto-Embedding**: Transparently embed documents at index time.

### Developer Experience
- **Deterministic Vector Math**: Memory-layout-independent cosine scoring with SIMD-aware accumulation and min-heap top-K selection.
- **Query Diagnostics**: Detailed timing breakdown for every pipeline stage.
- **Pluggable Providers**: Easy integration with any embedding model or API.
- **CLI Tool**: Powerful terminal interface for indexing and querying.

---

## Packages

| Package | Description |
|---------|-------------|
| [`Retrievo`](https://www.nuget.org/packages/Retrievo) | Core library — BM25 lexical search, brute-force vector search, RRF fusion, builder, mutable index. Zero external service dependencies. |
| [`Retrievo.AzureOpenAI`](https://www.nuget.org/packages/Retrievo.AzureOpenAI) | Azure OpenAI embedding provider. Install this if you want automatic document/query embedding via Azure OpenAI. Adds a dependency on `Azure.AI.OpenAI`. |

```shell
dotnet add package Retrievo --prerelease
dotnet add package Retrievo.AzureOpenAI --prerelease  # optional, for Azure OpenAI embeddings
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

### Field Definitions

Declare field types at index time so filters automatically use the right matching strategy:

```csharp
using var index = new HybridSearchIndexBuilder()
    .DefineField("tags", FieldType.StringArray)         // pipe-delimited by default
    .DefineField("categories", FieldType.StringArray, delimiter: ',')
    .AddDocument(new Document
    {
        Id = "1",
        Body = "Deep learning fundamentals",
        Metadata = new Dictionary<string, string>
        {
            ["tags"] = "ml|deep-learning|neural-nets",
            ["categories"] = "ai,education"
        }
    })
    .Build();

// StringArray fields auto-split and do contains-match; undeclared fields use exact-match
var response = index.Search(new HybridQuery
{
    Text = "deep learning",
    MetadataFilters = new Dictionary<string, string> { ["tags"] = "ml" }
});
```

### Snapshot Export and Import

Persist a built index to a versioned JSON snapshot and reload it later without rescanning files or regenerating stored embeddings:

```csharp
using var index = await new HybridSearchIndexBuilder()
    .AddFolder("./docs")
    .WithEmbeddingProvider(provider)
    .BuildAsync();

await index.ExportSnapshotAsync("./docs.retrievo.json");

using var restored = await HybridSearchIndex.ImportSnapshotAsync(
    "./docs.retrievo.json",
    embeddingProvider: provider);
var response = await restored.SearchAsync(new HybridQuery { Text = "how to deploy", TopK = 5 });
```

Mutable indexes export the last committed snapshot only. Call `Commit()` before exporting pending changes.
Snapshots persist the live normalized vector state, so round-tripped vector rankings stay identical.
Stored document embeddings survive snapshot export/import, but text queries still need an embedding provider at import time if you want vector or hybrid retrieval after restore.

Built-in `RrfFuser` snapshots import directly. If you use a custom `IFuser`, supply the same fuser again during import so ranking semantics are preserved:

```csharp
using var restored = HybridSearchIndex.ImportSnapshot(
    "./docs.retrievo.json",
    fuser: new MyCustomFuser());
```

The CLI supports built-in `RrfFuser` snapshots directly.

```shell
retrievo export ./docs --output ./docs.retrievo.json
retrievo query ./docs.retrievo.json --text "neural network"
```

For end-to-end snapshot parity checks against the checked-in BEIR benchmark fixtures:

```shell
dotnet run --project benchmarks/Retrievo.Benchmarks -- --dataset nfcorpus --embeddings benchmarks/fixtures/embeddings/nfcorpus.text-embedding-3-small.cache --verify-snapshot-roundtrip
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

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | Done | MVP hybrid retrieval, CLI, Azure OpenAI provider |
| **Phase 2** | Done | Mutable index, fielded search, filters (exact, range, contains), field definitions, diagnostics |
| **Phase 3** | Done | Snapshot export and import |
| **Phase 4** | Planned | ANN support for larger corpora |

---

## Build & Test

Requires [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later.

```shell
dotnet build
dotnet test
```

259 tests covering retrieval, vector math, fusion, mutable index, snapshot persistence, filters, field definitions, cancellation, and CLI integration — 0 warnings.
CLI integration tests build the CLI project and execute the matching built artifact for the active configuration/TFM, so `dotnet test tests/Retrievo.IntegrationTests` works from a clean checkout.

## Known Limitations

- **Lexical (BM25) search is English-only**: The lexical retrieval pipeline uses `EnglishStemAnalyzer` (StandardTokenizer → EnglishPossessiveFilter → LowerCaseFilter → English StopWords → PorterStemmer). Non-English text will not be properly tokenized or stemmed for BM25 matching.
- **Vector search is language-agnostic**: Semantic search works with any language supported by your embedding model (e.g., multilingual embeddings). Hybrid search inherits the English-only limitation for its lexical component.
- **Brute-force vector search is O(n) per query**: Designed for corpora up to ~10k documents. For larger corpora, consider ANN-based solutions (planned for Phase 4).
- **Manual persistence**: Indexes stay in-memory at runtime. Durability is explicit via snapshot export/import; there is no automatic crash recovery or write-ahead log.
- **No concurrent writers on `HybridSearchIndex`**: The immutable index is built once via the builder. Use `MutableHybridSearchIndex` for incremental upserts and deletes.
- **Single-process**: No distributed or shared index support. The index lives in a single process's memory.
- **Workaround for non-English corpora**: Use vector-only search by omitting lexical configuration, or configure a custom analyzer for your language in a fork.

---

## License

[MIT](https://github.com/TianqiZhang/Retrievo/blob/master/LICENSE)
