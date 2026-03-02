namespace HybridSearch.Models;

/// <summary>
/// Per-component timing breakdown for a search query.
/// All times are in milliseconds.
/// </summary>
public sealed record QueryTimingBreakdown
{
    /// <summary>
    /// Time spent in lexical (BM25) retrieval, or null if lexical search was not performed.
    /// </summary>
    public double? LexicalTimeMs { get; init; }

    /// <summary>
    /// Time spent in vector (cosine similarity) retrieval, or null if vector search was not performed.
    /// </summary>
    public double? VectorTimeMs { get; init; }

    /// <summary>
    /// Time spent in RRF fusion.
    /// </summary>
    public double FusionTimeMs { get; init; }

    /// <summary>
    /// Time spent embedding the query text, or null if no embedding was needed.
    /// </summary>
    public double? EmbeddingTimeMs { get; init; }

    /// <summary>
    /// Time spent applying metadata filters, or null if no filters were applied.
    /// </summary>
    public double? FilterTimeMs { get; init; }

    /// <summary>
    /// Total wall-clock query time including embedding, retrieval, fusion, and filtering.
    /// </summary>
    public double TotalTimeMs { get; init; }
}
