namespace HybridSearch.Models;

/// <summary>
/// A single result from a hybrid search query.
/// </summary>
public sealed record SearchResult
{
    /// <summary>
    /// The document ID that matched.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// The fused relevance score. Higher is more relevant.
    /// </summary>
    public required double Score { get; init; }

    /// <summary>
    /// Optional detailed score breakdown. Only populated when
    /// <see cref="HybridQuery.Explain"/> is true.
    /// </summary>
    public ExplainDetails? Explain { get; init; }
}
