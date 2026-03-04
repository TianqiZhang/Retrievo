# Retrievo — Suggestions

> Based on a first-principles code audit of every source file, test file, and public contract. Organized by priority.

---

## Critical (fix before stable v1)

### 1. ~~Eliminate `.GetAwaiter().GetResult()` in sync Search paths~~ ✅ DONE

**Where**: `HybridSearchIndex.Search()`, `MutableHybridSearchIndex.Search()`  
**Problem**: Blocks the thread pool. Deadlocks on single-threaded `SynchronizationContext` (WPF, WinForms, older ASP.NET). Already acknowledged as tech debt in AGENTS.md.  
**Fix**: ~~Remove sync `Search()` entirely, or make the embedding call eagerly cached so the sync path never hits async I/O. If backward compat is needed, keep sync overloads only for the no-embedding-provider case (which is already fully synchronous).~~ **Fixed**: Sync `Search()` and `Build()` now throw `InvalidOperationException` when embedding provider is configured and async work would be required. Sync paths remain fully functional for the no-embedding case. XML docs updated with `<exception>` tags.

### 2. ~~Document the English-only limitation prominently~~ ✅ DONE

**Where**: README.md, `EnglishStemAnalyzer.cs`  
**Problem**: The lexical pipeline (StandardTokenizer → EnglishPossessiveFilter → LowerCase → StopWords → PorterStem) is hard-coded to English. Non-English text silently degrades — no error, just poor recall. Users will waste hours debugging.  
**Fix**: ~~Add a "Limitations" section to README.~~ **Fixed**: Added `## Known Limitations` section to README documenting the `EnglishStemAnalyzer` pipeline, vector search language-agnosticism, and workaround for non-English corpora.

---

## High (strongly recommended)

### 3. ~~Guard against NaN/Infinity in vector inputs~~ ✅ DONE

**Where**: `VectorMath.DotProduct`, `BruteForceVectorRetriever`  
**Problem**: If an embedding provider returns vectors with NaN or Infinity values, they propagate silently through scoring and produce meaningless results. No validation at insert time or query time.  
**Fix**: ~~Add a fast validation pass in the builder's `AddDocument` and in `SearchAsync` for query embeddings.~~ **Fixed**: `BruteForceVectorRetriever` now validates all embeddings in `Add()`, `Update()`, and `Search()` via a `ValidateFiniteValues()` helper that throws `ArgumentException` for NaN or Infinity values. Validation rejects the input before any state mutation (dimensions, entries).

### 4. ~~Migrate off `RAMDirectory`~~ ⏭️ NOT APPLICABLE

**Where**: `LuceneLexicalRetriever.cs` line 73  
**Problem**: `RAMDirectory` is deprecated in Lucene.NET. It works today but may be removed in future versions.  
**Status**: `ByteBuffersDirectory` does not exist in Lucene.NET 4.8.0-beta00016 (it was introduced in Java Lucene 8.4.0, and Lucene.NET 4.8.x ports Java Lucene 4.8.x). `RAMDirectory` is the correct and only in-memory implementation for this version. Revisit when Lucene.NET ports a newer Java version.

### 5. ~~Add `CancellationToken` to brute-force vector search~~ ✅ DONE

**Where**: `BruteForceVectorRetriever.Search()`  
**Problem**: For indices near the ~10k doc target, brute-force iteration without cancellation support means callers cannot abort long searches. Not a problem at 100 docs; becomes one at 10k with 1536-dimensional vectors.  
**Fix**: ~~Accept `CancellationToken`, check `token.ThrowIfCancellationRequested()` every N iterations.~~ **Fixed**: Added `Search(float[], int, CancellationToken)` overload to `IVectorRetriever` and `BruteForceVectorRetriever`. Checks `ct.ThrowIfCancellationRequested()` every 256 iterations. `HybridSearchIndex` and `MutableHybridSearchIndex` propagate the token from `SearchAsync` through to the vector scan. `MutableHybridSearchIndex.SearchVectorSnapshot` also checks cancellation.

### 6. ~~Publish XML documentation with NuGet package~~ ✅ DONE

**Where**: `Directory.Build.props`, `.csproj` files  
**Problem**: The codebase has excellent XML docs on all public types, but unless `GenerateDocumentationFile=true` is set and the XML file ships with the NuGet package, IDE consumers won't see them.  
**Fix**: ~~Ensure `<GenerateDocumentationFile>true</GenerateDocumentationFile>` in `Directory.Build.props`.~~ **Fixed**: Added `<GenerateDocumentationFile>true</GenerateDocumentationFile>` to `Directory.Build.props`. The XML file is automatically included in NuGet packages by the SDK.

---

## Medium (quality improvements)

### 7. ~~Add a "Limitations & When Not to Use" section to README~~ ✅ DONE

Suggested content:
- English-only lexical pipeline (see #2)
- Brute-force vector search: O(n) per query — designed for corpora ≤ ~10k documents
- In-memory only: no persistence, no crash recovery (index must be rebuilt)
- No concurrent writers on `HybridSearchIndex` (use `MutableHybridSearchIndex` for mutations)
- Single-process: no distributed/shared index support

**Fixed**: Expanded the README "Known Limitations" section with all five items listed above.

### 8. ~~Add integration test for NaN/Infinity propagation~~ ✅ DONE

**Where**: Test project  
**Problem**: No test covers what happens when embeddings contain NaN. If guard (#3) is added, this test validates it. If not, this test documents the behavior.  
**Fixed**: `BruteForceVectorRetrieverTests` includes comprehensive NaN/Infinity tests: `Add_NaNEmbedding_Throws`, `Update_InfinityEmbedding_Throws`, `Search_NaNQueryVector_Throws`, `Add_InvalidFirstEmbedding_DoesNotSetDimensions`, `Update_InvalidFirstEmbedding_DoesNotSetDimensions`.

### 9. Consider `ReadOnlyMemory<float>` for embeddings

**Where**: `Document.Embedding`, `HybridQuery.Embedding`, `VectorMath.DotProduct`  
**Problem**: Current API uses `float[]` which allows callers to mutate the array after insertion, potentially corrupting the normalized vector cache. Using `ReadOnlyMemory<float>` makes the immutability contract explicit.  
**Trade-off**: This is a public API change. Consider for v1.0 rather than patching now.

### 10. ~~Structured CLI error handling~~ ✅ DONE

**Where**: `Program.cs` (CLI)  
**Problem**: AGENTS.md already flags `ex.Message.Contains("...")` as an anti-pattern. The CLI catches exceptions by string-matching error messages.  
**Fix**: ~~Use specific exception types in catch blocks.~~ **Fixed**: Replaced `catch (InvalidOperationException ex) when (ex.Message.Contains("no documents"))` with `catch (InvalidOperationException)`. The only `InvalidOperationException` thrown from `BuildAsync` is the no-documents case, making string matching unnecessary.

---

## Low (nice to have)

### 11. Snapshot export/import (already on roadmap)

Phase 3 on the roadmap. Would enable persisting and restoring indices without rebuilding. Important for production use cases where rebuild time matters.

### 12. ANN support for larger corpora (already on roadmap)

Phase 4 on the roadmap. HNSW or similar would extend the useful range from ~10k to ~1M documents. The current brute-force approach is the right default for the target scale.

### 13. Benchmarks in CI

**Where**: CI pipeline  
**Problem**: BEIR benchmarks exist in the repo but aren't run in CI. Regression in retrieval quality would go undetected.  
**Fix**: Add a benchmark job to CI that validates NDCG@10 stays above a threshold. Can be gated on manual trigger to avoid slowing PRs.

### 14. ~~Source Link for NuGet debugging~~ ✅ DONE

**Where**: `Directory.Build.props`  
**Problem**: Without Source Link, users debugging through Retrievo code in Visual Studio see decompiled IL instead of source.  
**Fixed**: `Directory.Build.props` already includes `<PublishRepositoryUrl>true</PublishRepositoryUrl>`, `<EmbedUntrackedSources>true</EmbedUntrackedSources>`, `<IncludeSymbols>true</IncludeSymbols>`, `<SymbolPackageFormat>snupkg</SymbolPackageFormat>`, and `<Deterministic>true</Deterministic>`. Source Link is built-in for .NET 8 SDK with GitHub repos.

---

## What's Already Great (don't change)

- **3 dependencies total** — exceptional minimalism for a hybrid search engine
- **SIMD-first vector math** with scalar fallback — correct and performant
- **BM25 parameters tuned to Anserini BEIR defaults** — research-informed, not cargo-culted
- **Builder pattern with internal constructors** — invalid state is unrepresentable
- **Deterministic tie-breaking everywhere** — reproducible results across runs
- **`TreatWarningsAsErrors`** — prevents quality drift
- **225 tests including BEIR validation** — rare for a library this small
- **Clean abstraction boundaries** — every interface is small and focused
- **No logging in library** — correct for a library; exceptions are the API
