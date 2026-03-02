using HybridSearch.Models;

namespace HybridSearch.Abstractions;

/// <summary>
/// Fuses multiple ranked lists into a single ranked list using a fusion strategy (e.g., RRF).
/// </summary>
public interface IFuser
{
    /// <summary>
    /// Fuse multiple ranked lists into a single ranked list.
    /// </summary>
    /// <param name="rankedLists">The ranked lists to fuse, each paired with its weight.</param>
    /// <param name="rrfK">The RRF constant k (typically 60).</param>
    /// <param name="topK">Maximum number of results to return.</param>
    /// <param name="explain">If true, populate <see cref="ExplainDetails"/> on each result.</param>
    /// <returns>A fused, ranked list of search results.</returns>
    IReadOnlyList<SearchResult> Fuse(
        IReadOnlyList<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)> rankedLists,
        int rrfK,
        int topK,
        bool explain);
}
