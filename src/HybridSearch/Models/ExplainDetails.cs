namespace HybridSearch.Models;

/// <summary>
/// Detailed score breakdown for a single search result.
/// Only populated when <see cref="HybridQuery.Explain"/> is true.
/// </summary>
public sealed record ExplainDetails
{
    /// <summary>
    /// The document's rank in the lexical result list (1-based), or null if
    /// the document was not returned by lexical retrieval.
    /// </summary>
    public int? LexicalRank { get; init; }

    /// <summary>
    /// The document's rank in the vector result list (1-based), or null if
    /// the document was not returned by vector retrieval.
    /// </summary>
    public int? VectorRank { get; init; }

    /// <summary>
    /// The RRF contribution from the lexical list: LexicalWeight * 1/(RrfK + rank), or 0 if absent.
    /// </summary>
    public double LexicalContribution { get; init; }

    /// <summary>
    /// The RRF contribution from the vector list: VectorWeight * 1/(RrfK + rank), or 0 if absent.
    /// </summary>
    public double VectorContribution { get; init; }

    /// <summary>
    /// Total fused score (sum of all contributions).
    /// </summary>
    public double FusedScore { get; init; }
}
