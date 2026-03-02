using HybridSearch.Models;

namespace HybridSearch.Abstractions;

/// <summary>
/// A mutable hybrid search index supporting incremental updates (Phase 2).
/// Safe for concurrent reads. Writes require external synchronization or single-writer discipline.
/// Updates are not visible to readers until <see cref="Commit"/> is called.
/// </summary>
public interface IMutableHybridSearchIndex : IHybridSearchIndex
{
    /// <summary>
    /// Insert or update a document. The change is not visible to readers until <see cref="Commit"/> is called.
    /// </summary>
    void Upsert(Document doc);

    /// <summary>
    /// Insert or update a document asynchronously. Required when the document needs embedding.
    /// The change is not visible to readers until <see cref="Commit"/> is called.
    /// </summary>
    Task UpsertAsync(Document doc, CancellationToken ct = default);

    /// <summary>
    /// Remove a document by ID. Returns true if the document was found and marked for deletion.
    /// The change is not visible to readers until <see cref="Commit"/> is called.
    /// </summary>
    bool Delete(string id);

    /// <summary>
    /// Atomically swap in a new snapshot containing all pending changes, making them visible to readers.
    /// </summary>
    void Commit();
}
