# HybridSearch.NET

An open-source, in-process, in-memory **hybrid retrieval** library for .NET. Combines BM25 lexical search, vector similarity search, and [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) to search small local document corpora without running external infrastructure.

## When to use this

- **Local agent memory** -- file-based notes, digests, conversation summaries
- **Small RAG** over private docs -- team SOPs, specs, runbooks
- **Developer tools** indexing local Markdown/text
- **Offline / edge** scenarios with strict locality requirements

Designed for corpora up to ~10k documents. If you need large-scale distributed search, use a dedicated search engine.

## Features

- **Hybrid search** -- lexical (BM25) + vector (cosine similarity) + RRF fusion in a single query
- **Lexical-only or vector-only** -- either component works standalone
- **Explain mode** -- per-result score breakdown showing lexical rank, vector rank, and RRF contributions
- **Fluent builder API** -- add documents programmatically or ingest from a folder of `.md`/`.txt` files
- **Mutable index** -- incremental upsert/delete with explicit `Commit()` visibility boundary
- **Fielded search** -- title and body fields with configurable boost weights
- **Metadata filters** -- exact-match key-value filters applied post-fusion
- **Query timing breakdown** -- per-component diagnostics (lexical, vector, fusion, embedding, filter)
- **Pluggable embeddings** -- bring your own `IEmbeddingProvider`; ships with an Azure OpenAI provider
- **Auto-embedding** -- documents without pre-computed embeddings are embedded automatically at build time
- **SIMD-accelerated** -- brute-force vector search uses hardware intrinsics where available
- **Thread-safe reads** -- built indexes are safe for concurrent queries
- **CLI tool** -- `hybridsearch query <folder> --text "..."` for quick searches from the terminal
- **Zero external dependencies at runtime** -- everything runs in-process (Lucene.NET for lexical, managed code for vector)

## Quick start

### Library usage

```csharp
using HybridSearch;
using HybridSearch.Models;

// Build an index from documents
var index = new HybridSearchIndexBuilder()
    .AddDocument(new Document
    {
        Id = "doc-1",
        Body = "Neural networks learn complex patterns from training data.",
        Embedding = new float[] { 0.1f, 0.2f, 0.3f, /* ... */ }
    })
    .AddDocument(new Document
    {
        Id = "doc-2",
        Body = "Kubernetes orchestrates containerized application deployments.",
        Embedding = new float[] { 0.4f, 0.5f, 0.6f, /* ... */ }
    })
    .Build();

using (index)
{
    var response = index.Search(new HybridQuery
    {
        Text = "neural network training",       // lexical component
        Vector = new float[] { 0.1f, 0.2f, 0.3f, /* ... */ }, // vector component
        TopK = 10
    });

    foreach (var result in response.Results)
        Console.WriteLine($"{result.Id}: {result.Score:F6}");
}
```

### Ingest from a folder

```csharp
using var index = new HybridSearchIndexBuilder()
    .AddFolder("/path/to/docs")  // loads *.md and *.txt recursively
    .Build();

var response = index.Search(new HybridQuery { Text = "deployment guide" });
```

### With Azure OpenAI embeddings

```csharp
using HybridSearch.AzureOpenAI;

var provider = new AzureOpenAIEmbeddingProvider(
    endpoint: new Uri("https://your-resource.openai.azure.com/"),
    apiKey: "your-api-key",
    deploymentName: "text-embedding-3-small"
);

// Documents without embeddings are auto-embedded at build time
using var index = await new HybridSearchIndexBuilder()
    .AddFolder("/path/to/docs")
    .WithEmbeddingProvider(provider)
    .BuildAsync();

// Query text is auto-embedded when no vector is provided
var response = await index.SearchAsync(new HybridQuery
{
    Text = "how to deploy to production",
    TopK = 5,
    Explain = true
});
```

### Explain mode

```csharp
var response = index.Search(new HybridQuery
{
    Text = "machine learning",
    Vector = queryEmbedding,
    TopK = 5,
    Explain = true
});

foreach (var r in response.Results)
{
    var ex = r.Explain!;
    Console.WriteLine($"{r.Id} (score: {r.Score:F6})");
    Console.WriteLine($"  Lexical: rank={ex.LexicalRank}, contrib={ex.LexicalContribution:F6}");
    Console.WriteLine($"  Vector:  rank={ex.VectorRank}, contrib={ex.VectorContribution:F6}");
    Console.WriteLine($"  Fused:   {ex.FusedScore:F6}");
}
```

### Custom weights

Shift the balance between lexical and vector retrieval:

```csharp
var response = index.Search(new HybridQuery
{
    Text = "kubernetes deployment",
    Vector = queryEmbedding,
    LexicalWeight = 0.3f,   // de-emphasize keyword matching
    VectorWeight = 1.5f,    // emphasize semantic similarity
    TopK = 10
});
```

## CLI

The CLI tool provides a quick way to search local document folders from the terminal.

```
dotnet run --project src/HybridSearch.Cli -- query <folder> --text "search terms" [options]
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--text` | `-t` | (required) | Search query text |
| `--top-k` | `-k` | 10 | Number of results to return |
| `--explain` | | false | Show per-result score breakdown |
| `--embedding-provider` | | auto-detect | Embedding provider (`azure-openai`) |

### Examples

```bash
# Lexical-only search (no embedding provider configured)
dotnet run --project src/HybridSearch.Cli -- query ./docs --text "deployment guide"

# With score breakdown
dotnet run --project src/HybridSearch.Cli -- query ./docs --text "neural networks" --explain -k 5

# With Azure OpenAI embeddings (set env vars first)
export HYBRIDSEARCH_AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export HYBRIDSEARCH_AZURE_OPENAI_KEY="your-api-key"
export HYBRIDSEARCH_AZURE_OPENAI_DEPLOYMENT="text-embedding-3-small"
dotnet run --project src/HybridSearch.Cli -- query ./docs --text "deployment guide"
```

### Sample output

```
Indexing files from: ./docs
Indexed 42 documents (lexical only) in 128.3ms

Found 5 results in 12.1ms:

  [1] file:deployment-guide.md  (score: 0.032787)
      lexical: rank=1 contrib=0.016393  vector: n/a  fused=0.032787
  [2] file:infrastructure.md    (score: 0.016129)
      lexical: rank=2 contrib=0.016129  vector: n/a  fused=0.016129
```

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
  (ranked results + optional explain)
```

### How RRF works

[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) combines multiple ranked lists without requiring score normalization:

```
score(doc) = sum over lists of: weight * 1 / (k + rank)
```

Where `k` is a constant (default 60) and `rank` is the document's 1-based position in each list. Documents appearing in both lexical and vector results get contributions from both, naturally boosting results that are relevant by multiple criteria.

## Project structure

```
src/
  HybridSearch/                    Core library
    Models/                        Document, HybridQuery, SearchResult, ExplainDetails, QueryTimingBreakdown
    Abstractions/                  IHybridSearchIndex, IMutableHybridSearchIndex, IEmbeddingProvider, IFuser, ...
    Fusion/                        RRF fusion implementation
    Vector/                        SIMD-accelerated brute-force cosine similarity
    Lexical/                       Lucene.NET BM25 lexical retrieval
    HybridSearchIndex.cs           Read-only batch-built orchestrator
    HybridSearchIndexBuilder.cs    Fluent builder API (batch)
    MutableHybridSearchIndex.cs    Mutable orchestrator with upsert/delete/commit
    MutableHybridSearchIndexBuilder.cs  Builder for mutable indexes

  HybridSearch.AzureOpenAI/        Azure OpenAI embedding provider
  HybridSearch.Cli/                CLI tool (System.CommandLine)

tests/
  HybridSearch.Tests/              Unit tests (185 tests)
  HybridSearch.IntegrationTests/   CLI integration tests (5 tests)
```

## Building and testing

Requires [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later.

```bash
# Restore and build
dotnet restore
dotnet build

# Run all tests
dotnet test

# Run only unit tests
dotnet test tests/HybridSearch.Tests

# Run only integration tests (includes CLI smoke tests)
dotnet test tests/HybridSearch.IntegrationTests
```

## Test coverage

190 tests covering:

| Category | Tests | What's covered |
|----------|-------|----------------|
| Models | 18 | Document, query, result immutability and equality |
| RRF Fusion | 11 | Single/multi-list fusion, weights, TopK, explain, tie-breaking |
| Vector Math | 7 | SIMD dot product, normalization, edge cases |
| Vector Retriever | 10 | Cosine similarity, ranking, TopK, empty index |
| Lexical Retriever | 16 | BM25 search, fielded search, case insensitivity, multi-term, TopK |
| Text Analyzer | 8 | Tokenization, case folding, empty input |
| Corpus Generators | 13 | Deterministic generation, topic clustering, edge cases |
| Index Builder | 12 | Fluent API, folder ingestion, embedding, validation |
| Index Orchestrator | 12 | Hybrid/lexical/vector-only search, explain, async, dispose |
| Mutable Index | 21 | Upsert, delete, commit visibility boundary, snapshot consistency |
| Fielded Search | 8 | Title/body boosts, zero-boost exclusion, mutable index boosts |
| Metadata Filters | 13 | Single/multi-filter, case sensitivity, null metadata, vector/hybrid filters |
| Query Timing | 10 | Per-component timing breakdown, async, mutable index |
| Explain Pipeline | 9 | End-to-end explain data flow through full pipeline |
| Performance Benchmarks | 6 | 3k docs x 768 dims: vector, lexical, hybrid, explain overhead |
| Azure OpenAI Provider | 11 | Constructor validation, interface contracts, batch/embed semantics |
| CLI Integration | 5 | Query smoke, explain output, top-k, error handling |

## Performance

Benchmarked on 3,000 documents with 768-dimensional embeddings (matching `text-embedding-3-small` output size):

| Operation | Avg time |
|-----------|----------|
| Index build (3k docs) | < 2s |
| Vector-only query | < 5ms |
| Lexical-only query | < 5ms |
| Hybrid query (lexical + vector + RRF) | < 10ms |
| Explain overhead | < 2x base query |

These are informational benchmarks, not hard SLAs. Actual performance depends on hardware, document size, and embedding dimensions.

## Roadmap

This library follows a phased development plan:

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | Done | Foundation: models, interfaces, project scaffold |
| **Phase 1** | Done | MVP hybrid retrieval, CLI, Azure OpenAI provider |
| **Phase 2** | Done | Incremental updates (upsert/delete/commit), fielded search with boosts, metadata filters, query timing breakdown |
| Phase 3 | Planned | Snapshot export/import, document chunking |
| Phase 4 | Planned | Approximate nearest neighbor (ANN) for larger corpora |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [Lucene.NET](https://github.com/apache/lucenenet) | 4.8.0-beta00016 | BM25 lexical search (planned for replacement with lighter-weight implementation) |
| [Azure.AI.OpenAI](https://www.nuget.org/packages/Azure.AI.OpenAI) | 2.1.0 | Azure OpenAI embeddings (optional, in `HybridSearch.AzureOpenAI`) |
| [System.CommandLine](https://github.com/dotnet/command-line-api) | 2.0.0-beta4 | CLI argument parsing (in `HybridSearch.Cli`) |

## License

MIT
