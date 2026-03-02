namespace HybridSearch.Models;

/// <summary>
/// A ranked item returned by a single retriever (lexical or vector) before fusion.
/// </summary>
public sealed record RankedItem
{
    /// <summary>
    /// The document ID.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// The raw score from the retriever (BM25 score or cosine similarity).
    /// </summary>
    public required double Score { get; init; }

    /// <summary>
    /// The 1-based rank in this retriever's result list.
    /// </summary>
    public required int Rank { get; init; }
}
