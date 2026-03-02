# HybridSearch.NET — Phased Spec

## 0. Purpose
HybridSearch.NET is an open-source, in-process, **in-memory hybrid retrieval** library for **small, local corpora** where running a full search backend is too heavy.

Typical usage:
- **Local agent memory** (file-based notes, digests, conversation summaries)
- **Small RAG** over private docs (team SOPs, specs, runbooks)
- Developer tools indexing local **Markdown/text**
- Offline / edge scenarios with strict locality requirements

Core retrieval model:
- **Lexical**: BM25 / keyword search
- **Vector**: cosine/dot similarity over embeddings
- **Fusion**: Reciprocal Rank Fusion (RRF)

Non-goals: large-scale search (>10k–50k docs), distributed indexing, semantic re-ranking (cross-encoders).

---

## 1. Functional requirements

### 1.1 Library API (baseline)
- Provide a .NET library usable in-process.
- Support search over a corpus of documents:
  - `Text` query only (lexical)
  - `Vector` query only (vector)
  - `Text + Vector` (hybrid with fusion)
- Return top-K results.
- Provide **explainability** option:
  - lexical rank (if present)
  - vector rank (if present)
  - fused score breakdown (RRF contributions)
- Provide basic operational metadata:
  - corpus size, embedding dimension, index build time, query timing breakdown.

### 1.2 Document model
- Each document MUST have:
  - stable `Id` (string)
  - one or more text fields (MVP can default to a single `body` field)
- Each document MAY have:
  - `title`
  - `metadata` (key-value)
  - a single embedding vector (MVP)
- **Nullable embeddings:** a document with no embedding (null vector) is valid. It participates in lexical search only and is excluded from vector retrieval. If an `IEmbeddingProvider` is configured at index-build time, documents without embeddings are embedded automatically from their `body` text.

### 1.3 Corpus ingestion
- MVP MUST support building an index from:
  - `IEnumerable<Document>`
  - a folder of local files (at least `*.md`, `*.txt`)
- Folder ingestion defaults:
  - filename → `title`
  - file contents → `body`
  - metadata includes at least `sourcePath` and `lastModified`

### 1.4 Index update model
HybridSearch.NET MUST support at least one of these usage models:

**Model A — Batch build (MVP):**
- Build index from a set of docs.
- Query is read-only.
- Updates are applied via full rebuild.

**Model B — Incremental (Phase 2+):**
- Support `Upsert(id)` and `Delete(id)`.
- Define a clear **visibility boundary**:
  - updates become visible after `Commit()` (preferred for determinism), OR
  - auto-refresh with `Flush()` for tests.

### 1.5 CLI tool
- Provide a CLI for common local-doc workflows.
- MVP CLI commands:
  - `index <folder>` (build in memory for interactive querying)
  - `query <folder> --text "..."` (build then query)
  - `query <folder> --text "..." --explain`
- Optional MVP CLI mode:
  - `--watch` to rebuild on file changes (debounced).

### 1.6 Fielded search (optional for MVP)
- Phase 2 MUST support basic fielded search:
  - at least `title` and `body`
  - field boosts (e.g., `titleBoost`, `bodyBoost`)

### 1.7 Language support policy
- Default experience targets English (reasonable tokenizer, lowercase, etc.).
- MUST NOT hardcode “English-only.”
- Text analysis MUST be pluggable so future multi-language analyzers can be added.

---

## 2. Phased roadmap (functional)

### Phase 0 — Foundation
- Core models, interfaces, orchestration.
- Minimal sample app wiring.

**Acceptance**
- Build passes; unit tests compile and run.

### Phase 1 — MVP Hybrid Retrieval (1k–5k docs)
- Batch build + query-only index.
- Folder ingestion for Markdown/text.
- Lexical retrieval (BM25).
- Vector retrieval (brute-force cosine/dot).
- RRF fusion + weights.
- Explain mode.
- CLI demo.

**Acceptance**
- Correctness tests for lexical-only, vector-only, hybrid.
- Deterministic fused ranking for fixed inputs.
- Interactive latency on typical dev machine for ~3k docs.

### Phase 2 — Incremental Updates + Fielded Search (up to ~10k)
- Optional mutable index with `Upsert/Delete`.
- Explicit `Commit()` visibility boundary.
- `title/body` fields with boosts.
- Basic metadata filters (exact match).
- Diagnostics: query time breakdown.

**Acceptance**
- Consistent read snapshots during updates.
- Integration tests proving visibility boundary.
- Usable performance at ~10k docs.

### Phase 3 — Optional Snapshot + Chunking
- Optional snapshot export/import (avoid re-embedding).
- Optional chunking strategy for long Markdown.
- Parent-child mapping (chunk → document).

**Acceptance**
- Reload works without embedding recomputation.
- Chunking improves retrieval on long docs.

### Phase 4 — Optional ANN
- Pluggable vector index interface.
- Brute-force remains default.
- Optional ANN implementation behind a flag.

---

## 3. Implementation details (design + engineering)

### 3.1 Proposed architecture
- Separate concerns into components:
  - `ILexicalRetriever`
  - `IVectorRetriever`
  - `IFuser`
  - `ITextAnalyzer`
  - optional `IEmbeddingProvider`

### 3.2 Lexical retrieval implementation
- MVP: **Lucene.NET** with in-memory (`RAMDirectory`) storage.
- Use BM25 scoring.
- Analyzer selection behind `ITextAnalyzer`.

> **Trade-off note — Lucene.NET as MVP expedient.**
> Lucene.NET provides a proven inverted index, BM25 scoring, and analyzers out of the box, which accelerates MVP delivery. However, it is a large dependency with its own memory model, segment lifecycle, and threading assumptions. Once Phase 1 is complete and test coverage is solid, we should evaluate replacing Lucene.NET with a purpose-built in-memory inverted index. The `ILexicalRetriever` abstraction exists specifically to make this swap feasible without changing the public API. Acceptance criteria for any replacement: all existing lexical and hybrid tests pass, and memory/perf characteristics are equal or better for target corpus sizes (≤10k docs).

### 3.3 Vector retrieval implementation (MVP)
- Store embeddings in memory as `float[]`.
- Pre-normalize vectors for cosine similarity.
- Brute-force scan for top-K:
  - SIMD via `System.Numerics.Vector<float>`
  - optional parallelization (toggle; avoid default thread overhead for small N).

### 3.4 Embedding provider

The library does **not** include a built-in embedding model. Embeddings are supplied via a pluggable `IEmbeddingProvider` interface.

```csharp
public interface IEmbeddingProvider
{
    Task<float[]> EmbedAsync(string text, CancellationToken ct = default);
    Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
    int Dimensions { get; }
}
```

**MVP provider: Azure OpenAI embeddings.**
- The library ships an `AzureOpenAIEmbeddingProvider` that calls the Azure OpenAI embedding endpoint (e.g., `text-embedding-3-small`).
- Configuration: endpoint URL, API key, model deployment name, and optional dimensions override.
- Batch support maps to the Azure OpenAI batch embedding API to minimize round-trips during ingestion.

**Usage contract:**
- If `IEmbeddingProvider` is supplied at index-build time, documents without a pre-computed vector will be embedded automatically.
- If `IEmbeddingProvider` is **not** supplied, documents MUST include a pre-computed `Embedding` or they will participate in lexical search only.
- Query-time embedding: when a `HybridQuery` has `Text` but no `Vector`, and an `IEmbeddingProvider` is available, the engine embeds the query text automatically before vector retrieval.
- When no `IEmbeddingProvider` is configured and no `Vector` is supplied on the query, the search degrades gracefully to lexical-only.

**Future providers (non-MVP):**
- Local ONNX runtime (offline / air-gapped scenarios).
- OpenAI (non-Azure) endpoint.
- Custom user-supplied implementations.

### 3.5 Fusion implementation
- RRF (Reciprocal Rank Fusion):
  - for each list rank `r`, add `weight * 1/(rrfK + r)`
  - sum contributions across lists
  - sort by fused score
- Deterministic tie-breakers:
  - stable ordering by `Id` when scores match.
  - `Id` comparison uses **ordinal** (case-sensitive, culture-invariant) ordering.

### 3.6 Incremental update strategy (Phase 2)
**Goal:** avoid partial visibility and keep tests deterministic.

Preferred design: **read snapshots + commit boundary**.

- Maintain an immutable `IndexSnapshot` for readers:
  - lexical searcher
  - embedding store snapshot
  - metadata snapshot
  - version
- Queries read the current snapshot atomically.
- Writers apply updates and then `Commit()` swaps in a new snapshot.

Why this matters:
- Readers never lock.
- No mixed-state results.
- Tests can assert “not visible until commit.”

### 3.7 CLI implementation
- Use a folder reader with basic frontmatter-aware parsing later (optional).
- Implement `--watch` via debounced file system events.
- For CLI testability:
  - prefer a file system abstraction (e.g., `System.IO.Abstractions`).

**CLI embedding configuration:**
- The CLI reads embedding provider settings from environment variables or a config file (e.g., `hybridsearch.json` in the target folder or user profile).
- Required settings for Azure OpenAI: `HYBRIDSEARCH_AZURE_OPENAI_ENDPOINT`, `HYBRIDSEARCH_AZURE_OPENAI_KEY`, `HYBRIDSEARCH_AZURE_OPENAI_DEPLOYMENT`.
- If no embedding provider is configured, the CLI operates in **lexical-only mode** and prints a warning.
- `--embedding-provider` flag allows explicit selection (e.g., `--embedding-provider azure-openai`).
- Embeddings are cached alongside the index to avoid re-embedding unchanged documents on subsequent runs.

### 3.8 Testing strategy
- Unit tests:
  - RRF correctness
  - vector similarity correctness
  - analyzer/tokenizer behavior via injectable analyzer
- Integration tests:
  - folder ingestion → build → query
  - incremental update visibility boundary (Phase 2)
- Perf sanity (non-gating):
  - synthetic corpora at 1k/5k/10k
  - record timing breakdown

---

## 4. Public API sketch (non-final)

```csharp
public interface IHybridSearchIndex : IDisposable
{
    SearchResponse Search(HybridQuery query);
    Task<SearchResponse> SearchAsync(HybridQuery query, CancellationToken ct = default);
    IndexStats GetStats();
}

public interface IMutableHybridSearchIndex : IHybridSearchIndex
{
    void Upsert(Document doc);
    Task UpsertAsync(Document doc, CancellationToken ct = default);
    bool Delete(string id);

    // Defines update visibility boundary
    void Commit();
}

public sealed record HybridQuery(
    string? Text,
    float[]? Vector,
    int TopK = 10,
    int LexicalK = 50,
    int VectorK = 50,
    float LexicalWeight = 1f,
    float VectorWeight = 1f,
    int RrfK = 60,
    bool Explain = false,
    Dictionary<string, string>? MetadataFilters = null);
```

> **Parameter defaults rationale:**
> - `TopK = 10` — standard default for retrieval; most callers want a short ranked list.
> - `LexicalK = 50`, `VectorK = 50` — candidate pool size per retriever. Larger than `TopK` so RRF has enough candidates to fuse meaningfully. 50 is a reasonable balance between recall and compute for small corpora.
> - `RrfK = 60` — the constant from the original RRF paper (Cormack, Clarke & Butt, 2009). Controls score decay; `60` is the standard value and works well empirically.
> - `LexicalWeight = 1f`, `VectorWeight = 1f` — equal weighting by default; caller adjusts to taste.
> - `MetadataFilters` — Phase 2. Exact-match key-value pairs applied as a post-retrieval filter. `null` means no filtering.

### 4.1 Async API note

Both sync and async methods are provided. The sync path is the natural fit for a pure in-memory index with no I/O. The async path exists because:
- `IEmbeddingProvider.EmbedAsync` is inherently async (network call to Azure OpenAI).
- `SearchAsync` needs to embed the query text when `Text` is provided but `Vector` is null and an embedding provider is configured.
- `UpsertAsync` may need to embed the document body if no pre-computed vector is supplied.

For the pure in-memory code path (no embedding call needed), the async methods may complete synchronously via `ValueTask` or similar. The sync methods will block on embedding if needed — callers in async contexts should prefer the async variants.

### 4.2 Thread safety contract

- `IHybridSearchIndex` (read-only, batch-built): **safe for concurrent reads** from multiple threads. No external synchronization required.
- `IMutableHybridSearchIndex`: **safe for concurrent reads**. Writes (`Upsert`, `Delete`, `Commit`) require **external synchronization** or single-writer discipline. Concurrent reads during a write cycle observe the last committed snapshot (never partial state).

---

## 5. Acceptance test checklist (by phase)

> The goal of this section is to make it easy to validate correctness and avoid regressions. These are phrased as *tests you can actually implement* (unit/integration), with clear pass/fail conditions.

### 5.1 Phase 0 — Foundation
**Unit tests**
- [ ] **Model immutability / equality:** `Document`, `HybridQuery`, and result models behave as expected (value equality where intended).
- [ ] **Deterministic ordering contract:** when fused scores tie, results are ordered by `Id` (or another explicitly documented stable tie-breaker).

**Integration tests**
- [ ] **Engine wiring smoke test:** create index from 3 documents and execute a query without exceptions.

---

### 5.2 Phase 1 — MVP Hybrid Retrieval
**Unit tests**
- [ ] **RRF correctness (single list):** given a ranked list `[A,B,C]`, fused scores follow `1/(k+r)` with correct ranks.
- [ ] **RRF correctness (two lists):** given lexical `[A,B,C]` and vector `[B,A,D]`, fused ranking matches expected ordering for a fixed `RrfK`, including weight handling.
- [ ] **Weighting behavior:** increasing `VectorWeight` shifts fused ordering in expected direction (e.g., doc that appears only in vector list rises).
- [ ] **Vector similarity:** cosine similarity returns 1.0 for identical vectors, ~0 for orthogonal vectors (within tolerance).
- [ ] **TopK truncation:** results length equals `TopK` (or `<=TopK` if corpus smaller).
- [ ] **Explain payload:** when `Explain=true`, each result includes lexical rank or null, vector rank or null, and fused contribution totals.

**Integration tests (folder ingestion)**
- [ ] **Index from folder:** building from a temp folder with `a.md`, `b.txt` produces 2 docs with expected `title/body` mapping.
- [ ] **Markdown ingestion sanity:** markdown file content is searchable (at least plain text extraction; no requirement for AST parsing).
- [ ] **Lexical-only query:** a keyword query retrieves the document containing that keyword in body.
- [ ] **Vector-only query:** a vector query retrieves the nearest document by cosine similarity using fixed test vectors.
- [ ] **Hybrid query:** with both text+vector, fused list contains union of candidates and returns expected top result.

**CLI tests**
- [ ] **CLI query smoke:** `hybridsearch query <folder> --text "..."` returns non-empty output and exit code 0.
- [ ] **CLI explain:** `--explain` prints lexical/vector ranks for returned docs.

**Perf sanity (non-gating, but tracked)**
- [ ] **3k docs scan:** vector brute-force for 3k docs (e.g., 768 dims) completes within a reasonable interactive budget on CI machine; record timing (do not hardcode strict SLA).

---

### 5.3 Phase 2 — Incremental Updates + Fielded Search
**Unit tests**
- [ ] **Visibility boundary (commit model):** after `Upsert()` but before `Commit()`, searches do **not** reflect the change.
- [ ] **Commit applies:** after `Commit()`, searches reflect the upserted/deleted doc.
- [ ] **Delete semantics:** after delete+commit, doc never appears even if it would match text/vector.
- [ ] **Field boosts:** when the query term appears in `title` only, increasing `titleBoost` increases rank relative to body-only matches.
- [ ] **Metadata filter include/exclude:** exact-match filter returns only matching docs.

**Integration tests**
- [ ] **Snapshot consistency:** during a writer update cycle, concurrent queries never throw and never return mixed-state artifacts (e.g., doc present in lexical results but missing from metadata store).
- [ ] **Incremental folder watcher (optional):** modifying a file triggers an upsert + commit and changes appear after the debounced cycle.

**Perf sanity**
- [ ] **10k docs scan:** vector brute-force query over ~10k docs remains usable; record timing and memory.

---

### 5.4 Phase 3 — Snapshot + Chunking (optional)
**Integration tests**
- [ ] **Snapshot round-trip:** build → export snapshot → import → query results match (within deterministic tie-breaker) for a fixed corpus.
- [ ] **No re-embed on import:** import does not call `IEmbeddingProvider` (assert via mock).
- [ ] **Chunk retrieval:** when chunking enabled, a match in the middle of a long doc returns the correct chunk and maps back to parent doc.

---

### 5.5 Phase 4 — ANN (optional)
**Correctness tests**
- [ ] **ANN parity check:** for a fixed small corpus, ANN top-K overlaps brute-force top-K above a configured threshold (e.g., recall@K).
- [ ] **Fallback:** brute-force remains functional and selectable.

---

## 6. Open questions
- Default analyzer selection and configuration (StandardAnalyzer likely sufficient for MVP).
- Snapshot format for Phase 3 (if implemented).
- ANN library selection (Phase 4).
- Chunking strategy (Phase 3): headings vs fixed-size tokens.
- Lucene.NET replacement timeline: evaluate after Phase 1 test coverage is solid; criteria documented in §3.2.
- Azure OpenAI embedding model selection: `text-embedding-3-small` (1536 dims) vs `text-embedding-3-large` (3072 dims) — smaller is likely sufficient for target corpus sizes and cheaper.
- Whether `IDisposable` is needed on the read-only index if Lucene.NET is eventually replaced with a pure managed implementation.

