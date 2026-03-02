using HybridSearch.Models;

namespace HybridSearch.Abstractions;

/// <summary>
/// A read-only hybrid search index supporting lexical, vector, and fused queries.
/// Thread-safe for concurrent reads.
/// </summary>
public interface IHybridSearchIndex : IDisposable
{
    /// <summary>
    /// Execute a hybrid search query synchronously.
    /// If the query requires embedding (Text provided, no Vector, and an embedding provider is configured),
    /// this method will block on the embedding call.
    /// </summary>
    SearchResponse Search(HybridQuery query);

    /// <summary>
    /// Execute a hybrid search query asynchronously.
    /// Preferred when an <see cref="IEmbeddingProvider"/> is configured and may need to embed query text.
    /// </summary>
    Task<SearchResponse> SearchAsync(HybridQuery query, CancellationToken ct = default);

    /// <summary>
    /// Returns operational statistics about the index.
    /// </summary>
    IndexStats GetStats();
}
