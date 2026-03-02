using HybridSearch.Models;
using HybridSearch.Tests.TestData;

namespace HybridSearch.Tests;

/// <summary>
/// End-to-end tests for explain mode wiring through the full pipeline:
/// HybridSearchIndex -> lexical + vector retrieval -> RRF fusion -> ExplainDetails.
/// Verifies that explain data flows correctly from retrievers through fusion to the response.
/// </summary>
public class ExplainPipelineTests
{
    [Fact]
    public void ExplainMode_HybridQuery_AllResultsHaveExplainDetails()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            Vector = docs[0].Embedding,
            TopK = 5,
            Explain = true
        });

        Assert.True(response.Results.Count > 0);
        Assert.All(response.Results, r =>
        {
            Assert.NotNull(r.Explain);
            Assert.Equal(r.Score, r.Explain!.FusedScore, precision: 10);
        });
    }

    [Fact]
    public void ExplainMode_False_NoExplainDetails()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            Vector = docs[0].Embedding,
            TopK = 5,
            Explain = false
        });

        Assert.True(response.Results.Count > 0);
        Assert.All(response.Results, r => Assert.Null(r.Explain));
    }

    [Fact]
    public void ExplainMode_LexicalOnly_LexicalRankPopulated_VectorRankNull()
    {
        // Use medium corpus to guarantee multiple documents contain "neural network"
        var docs = SyntheticCorpusGenerator.GenerateMedium(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            Vector = null, // lexical only
            TopK = 5,
            Explain = true
        });

        Assert.True(response.Results.Count > 0);
        Assert.All(response.Results, r =>
        {
            Assert.NotNull(r.Explain);
            Assert.NotNull(r.Explain!.LexicalRank);
            Assert.Null(r.Explain.VectorRank);
            Assert.True(r.Explain.LexicalContribution > 0);
            Assert.Equal(0.0, r.Explain.VectorContribution);
        });
    }

    [Fact]
    public void ExplainMode_VectorOnly_VectorRankPopulated_LexicalRankNull()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = null, // vector only
            Vector = docs[0].Embedding,
            TopK = 5,
            Explain = true
        });

        Assert.True(response.Results.Count > 0);
        Assert.All(response.Results, r =>
        {
            Assert.NotNull(r.Explain);
            Assert.Null(r.Explain!.LexicalRank);
            Assert.NotNull(r.Explain.VectorRank);
            Assert.Equal(0.0, r.Explain.LexicalContribution);
            Assert.True(r.Explain.VectorContribution > 0);
        });
    }

    [Fact]
    public void ExplainMode_HybridQuery_ContributionsSumToFusedScore()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "kubernetes deployment container",
            Vector = docs[0].Embedding,
            TopK = 10,
            Explain = true
        });

        Assert.True(response.Results.Count > 0);
        foreach (var result in response.Results)
        {
            Assert.NotNull(result.Explain);
            double expectedSum = result.Explain!.LexicalContribution + result.Explain.VectorContribution;
            Assert.Equal(expectedSum, result.Explain.FusedScore, precision: 10);
        }
    }

    [Fact]
    public void ExplainMode_Ranks_Are1Based()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "machine learning algorithms",
            Vector = docs[0].Embedding,
            TopK = 10,
            Explain = true
        });

        foreach (var result in response.Results)
        {
            Assert.NotNull(result.Explain);
            if (result.Explain!.LexicalRank.HasValue)
                Assert.True(result.Explain.LexicalRank.Value >= 1);
            if (result.Explain.VectorRank.HasValue)
                Assert.True(result.Explain.VectorRank.Value >= 1);
        }
    }

    [Fact]
    public void ExplainMode_FusedScore_MatchesResultScore()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "encryption security",
            Vector = docs[3].Embedding,
            TopK = 5,
            Explain = true
        });

        foreach (var result in response.Results)
        {
            Assert.NotNull(result.Explain);
            Assert.Equal(result.Score, result.Explain!.FusedScore, precision: 10);
        }
    }

    [Fact]
    public void ExplainMode_CustomWeights_AffectsContributions()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        // Equal weights
        var equalResponse = index.Search(new HybridQuery
        {
            Text = "distributed computing",
            Vector = docs[0].Embedding,
            TopK = 10,
            LexicalWeight = 1f,
            VectorWeight = 1f,
            Explain = true
        });

        // Heavy vector weight
        var vectorHeavyResponse = index.Search(new HybridQuery
        {
            Text = "distributed computing",
            Vector = docs[0].Embedding,
            TopK = 10,
            LexicalWeight = 0.1f,
            VectorWeight = 2f,
            Explain = true
        });

        // With heavy vector weight, the vector contribution should be proportionally larger
        if (vectorHeavyResponse.Results.Count > 0 && vectorHeavyResponse.Results[0].Explain is not null)
        {
            var explain = vectorHeavyResponse.Results[0].Explain!;
            if (explain.VectorRank.HasValue && explain.LexicalRank.HasValue)
            {
                // Vector contribution should dominate with 2.0 weight vs 0.1 weight
                Assert.True(explain.VectorContribution > explain.LexicalContribution,
                    $"Expected vector contribution ({explain.VectorContribution:F6}) > lexical ({explain.LexicalContribution:F6})");
            }
        }
    }

    [Fact]
    public async Task ExplainMode_WorksWithSearchAsync()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = await index.SearchAsync(new HybridQuery
        {
            Text = "neural network",
            Vector = docs[0].Embedding,
            TopK = 5,
            Explain = true
        });

        Assert.True(response.Results.Count > 0);
        Assert.All(response.Results, r =>
        {
            Assert.NotNull(r.Explain);
            Assert.Equal(r.Score, r.Explain!.FusedScore, precision: 10);
        });
    }
}
