using HybridSearch.Models;
using HybridSearch.Tests.TestData;

namespace HybridSearch.Tests;

public class QueryTimingTests
{
    [Fact]
    public void TimingBreakdown_IsPopulated_OnLexicalSearch()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            TopK = 5
        });

        Assert.NotNull(response.TimingBreakdown);
        var timing = response.TimingBreakdown;

        Assert.NotNull(timing.LexicalTimeMs);
        Assert.True(timing.LexicalTimeMs >= 0);
        Assert.Null(timing.VectorTimeMs); // No vector query
        Assert.Null(timing.EmbeddingTimeMs); // No embedding
        Assert.True(timing.FusionTimeMs >= 0);
        Assert.True(timing.TotalTimeMs >= 0);
    }

    [Fact]
    public void TimingBreakdown_IsPopulated_OnVectorSearch()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Vector = docs[0].Embedding,
            TopK = 5
        });

        Assert.NotNull(response.TimingBreakdown);
        var timing = response.TimingBreakdown;

        Assert.Null(timing.LexicalTimeMs); // No text query
        Assert.NotNull(timing.VectorTimeMs);
        Assert.True(timing.VectorTimeMs >= 0);
        Assert.Null(timing.EmbeddingTimeMs); // Vector provided directly
        Assert.True(timing.FusionTimeMs >= 0);
        Assert.True(timing.TotalTimeMs >= 0);
    }

    [Fact]
    public void TimingBreakdown_IsPopulated_OnHybridSearch()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            Vector = docs[0].Embedding,
            TopK = 5
        });

        Assert.NotNull(response.TimingBreakdown);
        var timing = response.TimingBreakdown;

        Assert.NotNull(timing.LexicalTimeMs);
        Assert.True(timing.LexicalTimeMs >= 0);
        Assert.NotNull(timing.VectorTimeMs);
        Assert.True(timing.VectorTimeMs >= 0);
        Assert.True(timing.FusionTimeMs >= 0);
        Assert.Null(timing.EmbeddingTimeMs); // Vector provided directly
        Assert.True(timing.TotalTimeMs >= 0);
    }

    [Fact]
    public void TimingBreakdown_FilterTimeMs_PopulatedWhenFiltersApplied()
    {
        using var index = new HybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "Neural networks learn complex patterns.",
                Metadata = new Dictionary<string, string> { ["topic"] = "ML" }
            })
            .AddDocument(new Document
            {
                Id = "doc-2",
                Body = "Kubernetes orchestrates containerized applications.",
                Metadata = new Dictionary<string, string> { ["topic"] = "Cloud" }
            })
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network kubernetes",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["topic"] = "ML" }
        });

        Assert.NotNull(response.TimingBreakdown);
        Assert.NotNull(response.TimingBreakdown.FilterTimeMs);
        Assert.True(response.TimingBreakdown.FilterTimeMs >= 0);
    }

    [Fact]
    public void TimingBreakdown_FilterTimeMs_NullWhenNoFilters()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            TopK = 5
        });

        Assert.NotNull(response.TimingBreakdown);
        Assert.Null(response.TimingBreakdown.FilterTimeMs);
    }

    [Fact]
    public void TimingBreakdown_EmptyQuery_StillPopulated()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = null,
            Vector = null,
            TopK = 5
        });

        Assert.NotNull(response.TimingBreakdown);
        Assert.True(response.TimingBreakdown.TotalTimeMs >= 0);
        // No components were invoked
        Assert.Null(response.TimingBreakdown.LexicalTimeMs);
        Assert.Null(response.TimingBreakdown.VectorTimeMs);
        Assert.Null(response.TimingBreakdown.EmbeddingTimeMs);
        Assert.Null(response.TimingBreakdown.FilterTimeMs);
    }

    [Fact]
    public void TimingBreakdown_TotalTimeMs_MatchesQueryTimeMs()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            Vector = docs[0].Embedding,
            TopK = 5
        });

        Assert.NotNull(response.TimingBreakdown);
        // TotalTimeMs and QueryTimeMs should be the same value
        Assert.Equal(response.QueryTimeMs, response.TimingBreakdown.TotalTimeMs);
    }

    [Fact]
    public void TimingBreakdown_MutableIndex_IsPopulated()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "doc-1",
            Body = "Neural networks learn complex patterns from training data.",
            Embedding = new float[] { 0.6f, 0.8f }
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            Vector = new float[] { 0.6f, 0.8f },
            TopK = 5
        });

        Assert.NotNull(response.TimingBreakdown);
        Assert.NotNull(response.TimingBreakdown.LexicalTimeMs);
        Assert.NotNull(response.TimingBreakdown.VectorTimeMs);
        Assert.True(response.TimingBreakdown.TotalTimeMs >= 0);
    }

    [Fact]
    public async Task TimingBreakdown_SearchAsync_IsPopulated()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = await index.SearchAsync(new HybridQuery
        {
            Text = "neural network",
            Vector = docs[0].Embedding,
            TopK = 5
        });

        Assert.NotNull(response.TimingBreakdown);
        Assert.NotNull(response.TimingBreakdown.LexicalTimeMs);
        Assert.NotNull(response.TimingBreakdown.VectorTimeMs);
        Assert.True(response.TimingBreakdown.TotalTimeMs >= 0);
    }

    [Fact]
    public void TimingBreakdown_ComponentTimesArePlausible()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium(embeddingDims: 128);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            Vector = docs[0].Embedding,
            TopK = 10
        });

        Assert.NotNull(response.TimingBreakdown);
        var timing = response.TimingBreakdown;

        // Each component time should be less than total
        Assert.True(timing.LexicalTimeMs!.Value <= timing.TotalTimeMs + 1); // +1ms tolerance
        Assert.True(timing.VectorTimeMs!.Value <= timing.TotalTimeMs + 1);
        Assert.True(timing.FusionTimeMs <= timing.TotalTimeMs + 1);
    }
}
