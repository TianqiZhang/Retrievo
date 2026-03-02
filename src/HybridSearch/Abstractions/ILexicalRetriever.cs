using HybridSearch.Models;

namespace HybridSearch.Abstractions;

/// <summary>
/// Performs lexical (keyword/BM25) retrieval over the indexed corpus.
/// </summary>
public interface ILexicalRetriever : IDisposable
{
    /// <summary>
    /// Search the lexical index for the given text query, returning up to <paramref name="topK"/> ranked results.
    /// </summary>
    /// <param name="text">The text query.</param>
    /// <param name="topK">Maximum number of results to return.</param>
    /// <returns>A ranked list of document IDs with scores, ordered by descending relevance.</returns>
    IReadOnlyList<RankedItem> Search(string text, int topK);
}
