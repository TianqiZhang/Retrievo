using System.Runtime.InteropServices;
using Retrievo.Abstractions;
using Retrievo.Models;

namespace Retrievo.Fusion;

/// <summary>
/// Implements Reciprocal Rank Fusion (RRF) as described by Cormack, Clarke &amp; Butt (2009).
/// For each document across all ranked lists, the fused score is:
///   sum over lists of: weight * 1/(rrfK + rank)
/// where rank is 1-based.
/// Ties are broken by ordinal (case-sensitive) comparison of document IDs.
/// </summary>
public sealed class RrfFuser : IFuser
{
    /// <inheritdoc/>
    public IReadOnlyList<SearchResult> Fuse(
        IReadOnlyList<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)> rankedLists,
        int rrfK,
        int topK,
        bool explain)
    {
        if (rankedLists.Count == 0)
            return Array.Empty<SearchResult>();

        // Accumulate scores and track per-list ranks for explain
        var scores = new Dictionary<string, double>(StringComparer.Ordinal);
        var explainData = explain
            ? new Dictionary<string, Dictionary<string, (int Rank, double Contribution)>>(StringComparer.Ordinal)
            : null;

        for (int listIdx = 0; listIdx < rankedLists.Count; listIdx++)
        {
            var (items, weight, listName) = rankedLists[listIdx];

            for (int i = 0; i < items.Count; i++)
            {
                var item = items[i];
                int rank = item.Rank; // 1-based
                double contribution = weight * (1.0 / (rrfK + rank));

                ref double scoreRef = ref CollectionsMarshal.GetValueRefOrAddDefault(scores, item.Id, out _);
                scoreRef += contribution;

                if (explainData is not null)
                {
                    if (!explainData.TryGetValue(item.Id, out var listContributions))
                    {
                        listContributions = new Dictionary<string, (int, double)>(StringComparer.Ordinal);
                        explainData[item.Id] = listContributions;
                    }
                    listContributions[listName] = (rank, contribution);
                }
            }
        }

        // Sort by descending score, then ordinal ascending ID for deterministic tie-breaking
        var sorted = scores
            .OrderByDescending(kvp => kvp.Value)
            .ThenBy(kvp => kvp.Key, StringComparer.Ordinal)
            .Take(topK)
            .ToList();

        var results = new List<SearchResult>(sorted.Count);

        foreach (var (id, score) in sorted)
        {
            ExplainDetails? explainDetails = null;

            if (explainData is not null)
            {
                int? lexicalRank = null;
                int? vectorRank = null;
                double lexicalContribution = 0.0;
                double vectorContribution = 0.0;

                if (explainData.TryGetValue(id, out var contributions))
                {
                    if (contributions.TryGetValue("lexical", out var lexInfo))
                    {
                        lexicalRank = lexInfo.Rank;
                        lexicalContribution = lexInfo.Contribution;
                    }
                    if (contributions.TryGetValue("vector", out var vecInfo))
                    {
                        vectorRank = vecInfo.Rank;
                        vectorContribution = vecInfo.Contribution;
                    }
                }

                explainDetails = new ExplainDetails
                {
                    LexicalRank = lexicalRank,
                    VectorRank = vectorRank,
                    LexicalContribution = lexicalContribution,
                    VectorContribution = vectorContribution,
                    FusedScore = score
                };
            }

            results.Add(new SearchResult
            {
                Id = id,
                Score = score,
                Explain = explainDetails
            });
        }

        return results;
    }
}
