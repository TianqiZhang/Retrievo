using Retrievo.Models;

namespace Retrievo.Abstractions;

/// <summary>
/// Performs vector (embedding similarity) retrieval over the indexed corpus.
/// </summary>
public interface IVectorRetriever
{
    /// <summary>
    /// Search for the nearest documents to the given vector, returning up to <paramref name="topK"/> ranked results.
    /// </summary>
    /// <param name="vector">The query embedding vector (must match index dimensions).</param>
    /// <param name="topK">Maximum number of results to return.</param>
    /// <returns>A ranked list of document IDs with cosine similarity scores, ordered by descending similarity.</returns>
    IReadOnlyList<RankedItem> Search(float[] vector, int topK);

    /// <summary>
    /// Search for the nearest documents to the given vector, returning up to <paramref name="topK"/> ranked results.
    /// Supports cooperative cancellation for large corpora.
    /// </summary>
    /// <param name="vector">The query embedding vector (must match index dimensions).</param>
    /// <param name="topK">Maximum number of results to return.</param>
    /// <param name="ct">Cancellation token for cooperative cancellation during the brute-force scan.</param>
    /// <returns>A ranked list of document IDs with cosine similarity scores, ordered by descending similarity.</returns>
    IReadOnlyList<RankedItem> Search(float[] vector, int topK, CancellationToken ct);
}
