using HybridSearch.Models;

namespace HybridSearch.Tests;

public class MetadataFilterTests
{
    private static HybridSearchIndex BuildIndexWithMetadata()
    {
        return new HybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "ml-1",
                Body = "Neural networks learn complex patterns from training data.",
                Metadata = new Dictionary<string, string>
                {
                    ["topic"] = "Machine Learning",
                    ["source"] = "synthetic",
                    ["level"] = "advanced"
                }
            })
            .AddDocument(new Document
            {
                Id = "ml-2",
                Body = "Gradient descent optimizes model parameters iteratively.",
                Metadata = new Dictionary<string, string>
                {
                    ["topic"] = "Machine Learning",
                    ["source"] = "manual",
                    ["level"] = "beginner"
                }
            })
            .AddDocument(new Document
            {
                Id = "cloud-1",
                Body = "Kubernetes orchestrates containerized application deployments.",
                Metadata = new Dictionary<string, string>
                {
                    ["topic"] = "Cloud Infrastructure",
                    ["source"] = "synthetic",
                    ["level"] = "advanced"
                }
            })
            .AddDocument(new Document
            {
                Id = "db-1",
                Body = "SQL queries retrieve structured data from relational tables.",
                Metadata = new Dictionary<string, string>
                {
                    ["topic"] = "Database Systems",
                    ["source"] = "manual",
                    ["level"] = "beginner"
                }
            })
            .AddDocument(new Document
            {
                Id = "no-meta",
                Body = "This document has no metadata at all.",
                Metadata = null
            })
            .Build();
    }

    [Fact]
    public void SingleFilter_ReturnsOnlyMatchingDocuments()
    {
        using var index = BuildIndexWithMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network gradient descent kubernetes SQL data",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["topic"] = "Machine Learning" }
        });

        Assert.All(response.Results, r =>
            Assert.True(r.Id == "ml-1" || r.Id == "ml-2",
                $"Expected only ML documents but got {r.Id}"));
    }

    [Fact]
    public void MultipleFilters_AppliesAndSemantics()
    {
        using var index = BuildIndexWithMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network gradient descent kubernetes SQL data",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string>
            {
                ["topic"] = "Machine Learning",
                ["source"] = "synthetic"
            }
        });

        // Only ml-1 has topic=Machine Learning AND source=synthetic
        Assert.Single(response.Results);
        Assert.Equal("ml-1", response.Results[0].Id);
    }

    [Fact]
    public void NoMatchingDocuments_ReturnsEmpty()
    {
        using var index = BuildIndexWithMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network gradient descent kubernetes SQL data",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["topic"] = "Nonexistent Topic" }
        });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void NullMetadataFilters_ReturnsAllMatchingDocuments()
    {
        using var index = BuildIndexWithMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network gradient descent kubernetes SQL data",
            TopK = 10,
            MetadataFilters = null
        });

        // Without filters, all documents with matching text should appear
        Assert.True(response.Results.Count > 0);
    }

    [Fact]
    public void EmptyMetadataFilters_ReturnsAllMatchingDocuments()
    {
        using var index = BuildIndexWithMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network gradient descent kubernetes SQL data",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string>()
        });

        // Empty filter dict should behave like no filters
        Assert.True(response.Results.Count > 0);
    }

    [Fact]
    public void DocumentWithNullMetadata_IsFilteredOut()
    {
        using var index = BuildIndexWithMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "document metadata",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["source"] = "synthetic" }
        });

        // "no-meta" document should not appear even though it matches the text
        Assert.DoesNotContain(response.Results, r => r.Id == "no-meta");
    }

    [Fact]
    public void FilterIsOrdinalCaseSensitive()
    {
        using var index = BuildIndexWithMetadata();

        // "machine learning" (lowercase) should NOT match "Machine Learning"
        var response = index.Search(new HybridQuery
        {
            Text = "neural network gradient descent kubernetes SQL data",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["topic"] = "machine learning" }
        });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void FilterKeyNotPresent_DocumentIsFilteredOut()
    {
        using var index = BuildIndexWithMetadata();

        // Filter on a key that doesn't exist in any document's metadata
        var response = index.Search(new HybridQuery
        {
            Text = "neural network gradient descent kubernetes SQL data",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["nonexistent-key"] = "value" }
        });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void TopK_RespectedAfterFiltering()
    {
        using var index = BuildIndexWithMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network gradient descent kubernetes SQL data",
            TopK = 1,
            MetadataFilters = new Dictionary<string, string> { ["topic"] = "Machine Learning" }
        });

        // Even though 2 ML docs exist, TopK=1 should limit results
        Assert.True(response.Results.Count <= 1);
    }

    [Fact]
    public void MetadataFilter_WorksWithMutableIndex()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "doc-1",
            Body = "Neural networks and deep learning research.",
            Metadata = new Dictionary<string, string> { ["topic"] = "ML" }
        });
        index.Upsert(new Document
        {
            Id = "doc-2",
            Body = "Neural networks applied to image classification.",
            Metadata = new Dictionary<string, string> { ["topic"] = "CV" }
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "neural networks",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["topic"] = "ML" }
        });

        Assert.Single(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
    }

    [Fact]
    public void MetadataFilter_WithVectorSearch()
    {
        var embedding1 = new float[] { 0.6f, 0.8f };
        var embedding2 = new float[] { 0.8f, 0.6f };

        using var index = new HybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "First document.",
                Embedding = embedding1,
                Metadata = new Dictionary<string, string> { ["category"] = "A" }
            })
            .AddDocument(new Document
            {
                Id = "doc-2",
                Body = "Second document.",
                Embedding = embedding2,
                Metadata = new Dictionary<string, string> { ["category"] = "B" }
            })
            .Build();

        var response = index.Search(new HybridQuery
        {
            Vector = embedding1,
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["category"] = "A" }
        });

        Assert.Single(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
    }

    [Fact]
    public void MetadataFilter_WithHybridSearch()
    {
        var embedding = new float[] { 0.6f, 0.8f };

        using var index = new HybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "ml-doc",
                Body = "Neural networks learn complex patterns.",
                Embedding = embedding,
                Metadata = new Dictionary<string, string> { ["topic"] = "ML" }
            })
            .AddDocument(new Document
            {
                Id = "cloud-doc",
                Body = "Neural networks deployed in cloud environments.",
                Embedding = new float[] { 0.8f, 0.6f },
                Metadata = new Dictionary<string, string> { ["topic"] = "Cloud" }
            })
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "neural networks",
            Vector = embedding,
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["topic"] = "ML" }
        });

        Assert.Single(response.Results);
        Assert.Equal("ml-doc", response.Results[0].Id);
    }

    [Fact]
    public void SyntheticCorpus_MetadataFilter_ByTopic()
    {
        var docs = TestData.SyntheticCorpusGenerator.GenerateSmall();

        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        // Filter to only Machine Learning docs
        var response = index.Search(new HybridQuery
        {
            Text = "neural network training",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["topic"] = "Machine Learning" }
        });

        // All returned docs should be ML docs
        foreach (var result in response.Results)
        {
            var doc = docs.First(d => d.Id == result.Id);
            Assert.Equal("Machine Learning", doc.Metadata!["topic"]);
        }
    }
}
