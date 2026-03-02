using System.Diagnostics;

namespace HybridSearch.Models;

/// <summary>
/// The response from a hybrid search query.
/// </summary>
public sealed record SearchResponse
{
    /// <summary>
    /// The ranked list of matching documents, ordered by descending fused score.
    /// </summary>
    public required IReadOnlyList<SearchResult> Results { get; init; }

    /// <summary>
    /// Total query execution time in milliseconds.
    /// </summary>
    public double QueryTimeMs { get; init; }

    /// <summary>
    /// Per-component timing breakdown for the query. Always populated.
    /// </summary>
    public QueryTimingBreakdown? TimingBreakdown { get; init; }
}
