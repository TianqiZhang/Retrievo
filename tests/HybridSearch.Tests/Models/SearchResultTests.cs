using HybridSearch.Models;

namespace HybridSearch.Tests.Models;

public class SearchResultTests
{
    [Fact]
    public void SearchResult_BasicProperties()
    {
        var result = new SearchResult { Id = "doc-1", Score = 0.95 };

        Assert.Equal("doc-1", result.Id);
        Assert.Equal(0.95, result.Score);
        Assert.Null(result.Explain);
    }

    [Fact]
    public void SearchResult_WithExplainDetails()
    {
        var explain = new ExplainDetails
        {
            LexicalRank = 1,
            VectorRank = 3,
            LexicalContribution = 0.0163,
            VectorContribution = 0.0159,
            FusedScore = 0.0322
        };

        var result = new SearchResult { Id = "doc-1", Score = 0.0322, Explain = explain };

        Assert.NotNull(result.Explain);
        Assert.Equal(1, result.Explain.LexicalRank);
        Assert.Equal(3, result.Explain.VectorRank);
    }

    [Fact]
    public void SearchResult_OrderByScoreDescending_ThenByIdOrdinal()
    {
        var results = new[]
        {
            new SearchResult { Id = "charlie", Score = 0.5 },
            new SearchResult { Id = "alpha", Score = 0.5 },
            new SearchResult { Id = "bravo", Score = 0.9 },
            new SearchResult { Id = "delta", Score = 0.5 },
        };

        // Sort by descending score, then ordinal ascending Id for tie-break
        var sorted = results
            .OrderByDescending(r => r.Score)
            .ThenBy(r => r.Id, StringComparer.Ordinal)
            .ToList();

        Assert.Equal("bravo", sorted[0].Id);    // highest score
        Assert.Equal("alpha", sorted[1].Id);     // tied at 0.5, ordinal: alpha < charlie < delta
        Assert.Equal("charlie", sorted[2].Id);
        Assert.Equal("delta", sorted[3].Id);
    }

    [Fact]
    public void SearchResult_OrdinalTieBreak_IsCaseSensitive()
    {
        var results = new[]
        {
            new SearchResult { Id = "Beta", Score = 0.5 },
            new SearchResult { Id = "alpha", Score = 0.5 },
        };

        var sorted = results
            .OrderByDescending(r => r.Score)
            .ThenBy(r => r.Id, StringComparer.Ordinal)
            .ToList();

        // Ordinal: uppercase letters come before lowercase in ASCII
        Assert.Equal("Beta", sorted[0].Id);
        Assert.Equal("alpha", sorted[1].Id);
    }

    [Fact]
    public void ExplainDetails_DocInOnlyOneList()
    {
        var explain = new ExplainDetails
        {
            LexicalRank = 2,
            VectorRank = null,
            LexicalContribution = 0.0161,
            VectorContribution = 0.0,
            FusedScore = 0.0161
        };

        Assert.Null(explain.VectorRank);
        Assert.Equal(0.0, explain.VectorContribution);
        Assert.Equal(explain.LexicalContribution, explain.FusedScore);
    }
}
