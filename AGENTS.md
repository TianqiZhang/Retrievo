# AGENTS.md — Retrievo

> Knowledge base for AI agents. Read this before making changes. See README.md for project overview and build commands.

---

## Workflow Rules (CRITICAL)

### 1. Update README on Every Change

Whenever features, tests, project structure, or the roadmap change, always make sure `README.md` is up-to-date.

### 2. Code Review Before Every Commit

Before committing any change, ask a **specialized code review expert agent** to review all pending changes. Do not bypass this step. 
---

## Coding Conventions

### Namespaces

Folder maps directly to namespace: `Retrievo`, `Retrievo.Abstractions`, `Retrievo.Models`, `Retrievo.Lexical`, `Retrievo.Vector`, `Retrievo.Fusion`.

Use file-scoped namespace declarations: `namespace Retrievo;`

### Types

| Pattern | When | Example |
|---------|------|---------|
| `public sealed record` | DTOs, value-like models | `HybridQuery`, `SearchResult`, `SearchResponse`, `IndexStats` |
| `public sealed class` | Orchestrators, mutable state | `HybridSearchIndex`, `MutableHybridSearchIndex`, `Document` |
| `internal` constructors | Force builder usage | `internal HybridSearchIndex(...)` |
| `public` builders | Fluent API for construction | `HybridSearchIndexBuilder`, `MutableHybridSearchIndexBuilder` |

### Properties

- Prefer `init`-only: `public string Name { get; init; }`
- Use `required` for mandatory fields: `public required string Id { get; init; }`
- Use `T?` for optional fields: `public string? Title { get; init; }`
- Expose collections as `IReadOnlyList<T>` or `IReadOnlyDictionary<K,V>`
- Use `float[]?` for embedding vectors

### Methods

- Provide both sync and async: `Search` + `SearchAsync`
- Async-first for I/O-bound operations (embedding providers)
- Keep helpers `private` or `internal`
- Propagate `CancellationToken` in async methods

### Interfaces (Abstractions/)

- Small, focused — single responsibility
- XML doc comments on all members
- Use `IReadOnlyList<T>` and `Task<T>` in signatures
- `IDisposable` on index types that hold resources

### XML Documentation

- **Required** on all public types, interfaces, and public methods
- Use `/// <inheritdoc/>` on implementations
- Include `<summary>`, `<param>`, `<returns>` where useful

### Imports

- File-scoped `using` statements
- Alias long external types: `using LuceneDocument = Lucene.Net.Documents.Document;`

---

## Validation & Error Handling

### Guard Clauses (DO THIS)

```csharp
// Null checks — use static throw helpers
ArgumentNullException.ThrowIfNull(query);
ArgumentNullException.ThrowIfNull(embedding);

// Disposed checks — at start of public methods
ObjectDisposedException.ThrowIf(_disposed, this);

// Numeric validation
if (float.IsNaN(query.TitleBoost) || float.IsInfinity(query.TitleBoost) || query.TitleBoost < 0)
    throw new ArgumentOutOfRangeException(nameof(query), "TitleBoost must be...");

// Dimension validation
if (embedding.Length != Dimensions)
    throw new ArgumentException($"Expected {Dimensions} dimensions, got {embedding.Length}");
```

### Exception Types

Use BCL exceptions only — no custom exception types in this codebase:
- `ArgumentNullException` — null inputs
- `ArgumentException` — invalid input values
- `ArgumentOutOfRangeException` — out-of-range numerics
- `InvalidOperationException` — invalid state (e.g., building with no documents)
- `ObjectDisposedException` — using disposed objects
- `DirectoryNotFoundException` — missing file paths

### Try/Catch (Narrow Catches Only)

```csharp
// DO: Catch specific, expected exceptions
catch (ParseException) { return Array.Empty<RankedItem>(); }

// DON'T: Broad catch-all or swallowing exceptions
catch (Exception) { }  // NEVER
```

### No Logging in Library

- No `ILogger` usage in core library — exceptions and return values are the API
- CLI uses `Console.Error.WriteLine` / `Console.WriteLine` for user-facing output only

---

## Resource Management

### IDisposable Pattern

- Use `_disposed` flag and guard with `ObjectDisposedException.ThrowIf`
- Dispose all owned resources: readers, writers, directories, analyzers
- Protect against double-dispose with flag check

### Lucene Ref-Counting (Mutable Index)

- `AcquireSearcherSnapshot()` calls `reader.IncRef()` — caller must release
- `ReleaseSearcherSnapshot(reader)` calls `reader.DecRef()`
- Snapshot swap on `Commit()` — old snapshot released, new one acquired

---

## Known Patterns & Anti-Patterns

### Patterns to Follow

- **Builder pattern** for index construction (fluent API → internal constructor)
- **Snapshot isolation** for mutable index (immutable snapshot swapped atomically)
- **Deterministic tie-breaking** in RRF: `OrderByDescending(score).ThenBy(id, StringComparer.Ordinal)`
- **Optional dependencies** via nullable: `IEmbeddingProvider?` with `if (provider is not null)` check
- **SIMD-first** vector math with scalar fallback in `VectorMath`

### Anti-Patterns to Avoid

| Anti-Pattern | Where It Exists | Why It's Bad |
|---|---|---|
| `.GetAwaiter().GetResult()` | `HybridSearchIndex.Search`, `MutableHybridSearchIndex.Search` | Blocks thread pool, potential deadlock. Known tech debt — do not add more. |
| `ex.Message.Contains("...")` | `Program.cs` (CLI) | Brittle — breaks if message text changes. Use specific exception types instead. |
| Type suppression (`as any` / `#pragma`) | Not present | Keep it that way. `TreatWarningsAsErrors` is on. |
| Empty catch blocks | Not present | Keep it that way. |
