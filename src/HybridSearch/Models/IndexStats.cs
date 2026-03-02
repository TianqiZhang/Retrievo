namespace HybridSearch.Models;

/// <summary>
/// Operational statistics about the search index.
/// </summary>
public sealed record IndexStats
{
    /// <summary>
    /// Total number of documents in the index.
    /// </summary>
    public int DocumentCount { get; init; }

    /// <summary>
    /// The embedding vector dimension, or null if no embeddings are present.
    /// </summary>
    public int? EmbeddingDimension { get; init; }

    /// <summary>
    /// Time taken to build the index in milliseconds.
    /// </summary>
    public double IndexBuildTimeMs { get; init; }
}
