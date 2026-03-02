using HybridSearch.Abstractions;
using HybridSearch.Models;
using HybridSearch.Tests.TestData;
using NSubstitute;

namespace HybridSearch.Tests;

public class HybridSearchIndexTests
{
    [Fact]
    public void LexicalOnlySearch_ReturnsMatchingDocuments()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var response = index.Search(new HybridQuery
            {
                Text = "neural network",
                TopK = 5
            });

            Assert.NotNull(response);
            Assert.NotNull(response.Results);
            Assert.True(response.QueryTimeMs >= 0);
        }
    }

    [Fact]
    public void VectorOnlySearch_ReturnsMatchingDocuments()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            // Use the first doc's embedding as query vector
            var response = index.Search(new HybridQuery
            {
                Vector = docs[0].Embedding,
                TopK = 5
            });

            Assert.NotNull(response);
            Assert.True(response.Results.Count > 0);
            // The first result should be the doc itself (identical embedding)
            Assert.Equal(docs[0].Id, response.Results[0].Id);
        }
    }

    [Fact]
    public void HybridSearch_CombinesLexicalAndVector()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var response = index.Search(new HybridQuery
            {
                Text = "neural network",
                Vector = docs[0].Embedding,
                TopK = 5
            });

            Assert.NotNull(response);
            Assert.True(response.Results.Count > 0);
        }
    }

    [Fact]
    public void GetStats_ReturnsCorrectCounts()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var stats = index.GetStats();

            Assert.Equal(5, stats.DocumentCount);
            Assert.Equal(128, stats.EmbeddingDimension);
            Assert.True(stats.IndexBuildTimeMs >= 0);
        }
    }

    [Fact]
    public void EmptyQuery_ReturnsEmptyResults()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var response = index.Search(new HybridQuery
            {
                Text = null,
                Vector = null,
                TopK = 5
            });

            Assert.Empty(response.Results);
        }
    }

    [Fact]
    public void NoMatchingTerms_ReturnsEmptyLexicalResults()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var response = index.Search(new HybridQuery
            {
                Text = "xyzzy12345nonexistent",
                TopK = 5
            });

            Assert.Empty(response.Results);
        }
    }

    [Fact]
    public void ExplainMode_PopulatesExplainDetails()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var response = index.Search(new HybridQuery
            {
                Text = "neural network",
                Vector = docs[0].Embedding,
                TopK = 5,
                Explain = true
            });

            // At least some results should have explain details
            var withExplain = response.Results.Where(r => r.Explain is not null).ToList();
            Assert.True(withExplain.Count > 0);
        }
    }

    [Fact]
    public async Task SearchAsync_Works()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var response = await index.SearchAsync(new HybridQuery
            {
                Text = "neural network",
                Vector = docs[0].Embedding,
                TopK = 5
            });

            Assert.NotNull(response);
            Assert.True(response.Results.Count > 0);
        }
    }

    [Fact]
    public async Task SearchAsync_WithEmbeddingProvider_EmbedsQueryText()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        var mockProvider = Substitute.For<IEmbeddingProvider>();
        mockProvider.Dimensions.Returns(128);
        mockProvider.EmbedAsync(Arg.Any<string>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(docs[0].Embedding!));
        mockProvider.EmbedBatchAsync(Arg.Any<IReadOnlyList<string>>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(Array.Empty<float[]>()));

        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .WithEmbeddingProvider(mockProvider)
            .Build();

        using (index)
        {
            // Query with text only — should auto-embed via provider
            var response = await index.SearchAsync(new HybridQuery
            {
                Text = "neural network",
                TopK = 5
            });

            Assert.NotNull(response);
            Assert.True(response.Results.Count > 0);

            // Verify embedding provider was called for the query
            await mockProvider.Received(1).EmbedAsync("neural network", Arg.Any<CancellationToken>());
        }
    }

    [Fact]
    public void TopK_IsRespected()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium(embeddingDims: 128);
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var response = index.Search(new HybridQuery
            {
                Vector = docs[0].Embedding,
                TopK = 3
            });

            Assert.True(response.Results.Count <= 3);
        }
    }

    [Fact]
    public void Dispose_ThenSearch_Throws()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => index.Search(new HybridQuery
        {
            Text = "test",
            TopK = 5
        }));
    }

    [Fact]
    public void MediumCorpus_SearchesSuccessfully()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium(embeddingDims: 128);
        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            var stats = index.GetStats();
            Assert.Equal(100, stats.DocumentCount);
            Assert.Equal(128, stats.EmbeddingDimension);

            // Hybrid search
            var response = index.Search(new HybridQuery
            {
                Text = "kubernetes deployment",
                Vector = docs[0].Embedding,
                TopK = 10
            });

            Assert.True(response.Results.Count > 0);
            Assert.True(response.Results.Count <= 10);

            // Results should be ordered by descending score
            for (int i = 1; i < response.Results.Count; i++)
                Assert.True(response.Results[i - 1].Score >= response.Results[i].Score);
        }
    }
}
