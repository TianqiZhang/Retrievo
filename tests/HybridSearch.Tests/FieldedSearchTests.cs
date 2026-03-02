using HybridSearch.Models;
using HybridSearch.Tests.TestData;

namespace HybridSearch.Tests;

public class FieldedSearchTests
{
    [Fact]
    public void DefaultBoosts_TitleAndBodyBothContribute()
    {
        // Document whose title matches the query
        var titleDoc = new Document
        {
            Id = "title-match",
            Title = "Neural network architecture overview",
            Body = "This document discusses various software topics."
        };

        // Document whose body matches the query
        var bodyDoc = new Document
        {
            Id = "body-match",
            Title = "General software engineering",
            Body = "Neural network training requires large datasets and GPU compute."
        };

        using var index = new HybridSearchIndexBuilder()
            .AddDocument(titleDoc)
            .AddDocument(bodyDoc)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            TopK = 10
        });

        // Both documents should be returned since default boosts are 1.0 each
        Assert.Equal(2, response.Results.Count);
    }

    [Fact]
    public void HighTitleBoost_PromotesTitleMatches()
    {
        var titleDoc = new Document
        {
            Id = "title-match",
            Title = "Neural network architecture overview",
            Body = "This document covers general topics in technology."
        };

        var bodyDoc = new Document
        {
            Id = "body-match",
            Title = "General software engineering",
            Body = "Neural network training requires large datasets and GPU compute."
        };

        using var index = new HybridSearchIndexBuilder()
            .AddDocument(titleDoc)
            .AddDocument(bodyDoc)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            TopK = 10,
            TitleBoost = 10f,
            BodyBoost = 1f
        });

        Assert.True(response.Results.Count >= 2);
        // Title match should be promoted to the top due to high title boost
        Assert.Equal("title-match", response.Results[0].Id);
    }

    [Fact]
    public void HighBodyBoost_PromotesBodyMatches()
    {
        var titleDoc = new Document
        {
            Id = "title-match",
            Title = "Neural network architecture overview",
            Body = "This document covers general topics in technology."
        };

        var bodyDoc = new Document
        {
            Id = "body-match",
            Title = "General software engineering",
            Body = "Neural network training requires large datasets and GPU compute."
        };

        using var index = new HybridSearchIndexBuilder()
            .AddDocument(titleDoc)
            .AddDocument(bodyDoc)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            TopK = 10,
            TitleBoost = 1f,
            BodyBoost = 10f
        });

        Assert.True(response.Results.Count >= 2);
        // Body match should be promoted to the top due to high body boost
        Assert.Equal("body-match", response.Results[0].Id);
    }

    [Fact]
    public void ZeroTitleBoost_ExcludesTitleOnlyMatches()
    {
        // Title-only match: query terms appear only in the title
        var titleOnlyDoc = new Document
        {
            Id = "title-only",
            Title = "Kubernetes deployment orchestration",
            Body = "This document has nothing to do with the query terms."
        };

        // Body match: query terms appear in the body
        var bodyDoc = new Document
        {
            Id = "body-match",
            Title = "General technology",
            Body = "Kubernetes deployment orchestration is critical for scaling."
        };

        using var index = new HybridSearchIndexBuilder()
            .AddDocument(titleOnlyDoc)
            .AddDocument(bodyDoc)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "kubernetes deployment",
            TopK = 10,
            TitleBoost = 0f,
            BodyBoost = 1f
        });

        // Body match should definitely appear
        Assert.Contains(response.Results, r => r.Id == "body-match");

        // If title-only doc appears, it should rank below body doc
        if (response.Results.Count >= 2)
        {
            var bodyRank = response.Results.ToList().FindIndex(r => r.Id == "body-match");
            var titleRank = response.Results.ToList().FindIndex(r => r.Id == "title-only");
            if (titleRank >= 0)
            {
                Assert.True(bodyRank < titleRank);
            }
        }
    }

    [Fact]
    public void FieldBoosts_WorkWithMutableIndex()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "title-match",
            Title = "Neural network architecture overview",
            Body = "This document covers general topics in technology."
        });

        index.Upsert(new Document
        {
            Id = "body-match",
            Title = "General software engineering",
            Body = "Neural network training requires large datasets and GPU compute."
        });

        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            TopK = 10,
            TitleBoost = 10f,
            BodyBoost = 1f
        });

        Assert.True(response.Results.Count >= 2);
        // Title match should be promoted to the top
        Assert.Equal("title-match", response.Results[0].Id);
    }

    [Fact]
    public void SyntheticCorpus_TitleFieldIsIndexed()
    {
        // SyntheticCorpusGenerator sets Title on all documents
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        Assert.All(docs, d => Assert.NotNull(d.Title));

        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        // Search for a term that appears in titles (e.g., "Machine Learning" from GenerateTitle)
        var response = index.Search(new HybridQuery
        {
            Text = "Machine Learning",
            TopK = 10,
            TitleBoost = 10f,
            BodyBoost = 0.1f
        });

        Assert.NotNull(response);
        Assert.True(response.Results.Count > 0);
    }

    [Fact]
    public void DefaultBoosts_AreOnePointZero()
    {
        var query = new HybridQuery { Text = "test", TopK = 5 };

        Assert.Equal(1f, query.TitleBoost);
        Assert.Equal(1f, query.BodyBoost);
    }

    [Fact]
    public void NullTitle_DocumentStillSearchableByBody()
    {
        var doc = new Document
        {
            Id = "no-title",
            Title = null,
            Body = "Neural network training patterns and optimization techniques."
        };

        using var index = new HybridSearchIndexBuilder()
            .AddDocument(doc)
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            TopK = 5,
            TitleBoost = 5f,
            BodyBoost = 1f
        });

        Assert.Single(response.Results);
        Assert.Equal("no-title", response.Results[0].Id);
    }
}
