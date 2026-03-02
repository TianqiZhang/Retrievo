using HybridSearch.Models;
using HybridSearch.Tests.TestData;

namespace HybridSearch.Tests;

public class MutableHybridSearchIndexTests
{
    [Fact]
    public void Build_EmptyIndex_CanSearchWithoutError()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        var response = index.Search(new HybridQuery { Text = "test", TopK = 5 });

        Assert.NotNull(response);
        Assert.Empty(response.Results);
    }

    [Fact]
    public void Upsert_ThenCommit_DocumentIsSearchable()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "doc-1",
            Body = "Neural networks learn complex patterns from training data."
        });
        index.Commit();

        var response = index.Search(new HybridQuery { Text = "neural network", TopK = 5 });

        Assert.Single(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
    }

    [Fact]
    public void Upsert_WithoutCommit_DocumentNotVisibleToReaders()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "doc-1",
            Body = "Neural networks learn complex patterns from training data."
        });

        // No Commit() — should not be visible
        var response = index.Search(new HybridQuery { Text = "neural network", TopK = 5 });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void Upsert_UpdateExistingDocument_ReflectsNewContent()
    {
        using var index = new MutableHybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "Old content about cats."
            })
            .Build();

        // Update the document
        index.Upsert(new Document
        {
            Id = "doc-1",
            Body = "New content about kubernetes deployment orchestration."
        });
        index.Commit();

        // Old content should not match
        var oldResponse = index.Search(new HybridQuery { Text = "cats", TopK = 5 });
        Assert.Empty(oldResponse.Results);

        // New content should match
        var newResponse = index.Search(new HybridQuery { Text = "kubernetes", TopK = 5 });
        Assert.Single(newResponse.Results);
        Assert.Equal("doc-1", newResponse.Results[0].Id);
    }

    [Fact]
    public void Delete_ExistingDocument_ReturnsTrue()
    {
        using var index = new MutableHybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "Neural networks learn complex patterns."
            })
            .Build();

        bool deleted = index.Delete("doc-1");

        Assert.True(deleted);
    }

    [Fact]
    public void Delete_NonexistentDocument_ReturnsFalse()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        bool deleted = index.Delete("nonexistent");

        Assert.False(deleted);
    }

    [Fact]
    public void Delete_ThenCommit_DocumentNotSearchable()
    {
        using var index = new MutableHybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "Neural networks learn complex patterns from training data."
            })
            .Build();

        // Verify initially searchable
        var before = index.Search(new HybridQuery { Text = "neural network", TopK = 5 });
        Assert.Single(before.Results);

        index.Delete("doc-1");
        index.Commit();

        var after = index.Search(new HybridQuery { Text = "neural network", TopK = 5 });
        Assert.Empty(after.Results);
    }

    [Fact]
    public void Delete_WithoutCommit_DocumentStillVisible()
    {
        using var index = new MutableHybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "Neural networks learn complex patterns from training data."
            })
            .Build();

        index.Delete("doc-1");

        // No Commit() — document should still be visible in the snapshot
        var response = index.Search(new HybridQuery { Text = "neural network", TopK = 5 });
        Assert.Single(response.Results);
    }

    [Fact]
    public void Commit_UpdatesStats()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        Assert.Equal(0, index.GetStats().DocumentCount);

        index.Upsert(new Document { Id = "doc-1", Body = "First document." });
        index.Upsert(new Document { Id = "doc-2", Body = "Second document." });
        index.Commit();

        Assert.Equal(2, index.GetStats().DocumentCount);

        index.Delete("doc-1");
        index.Commit();

        Assert.Equal(1, index.GetStats().DocumentCount);
    }

    [Fact]
    public void MultipleCommits_EachSnapshotIsConsistent()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        // First batch
        index.Upsert(new Document { Id = "doc-1", Body = "Alpha beta gamma." });
        index.Commit();

        var r1 = index.Search(new HybridQuery { Text = "alpha", TopK = 10 });
        Assert.Single(r1.Results);

        // Second batch — add more, delete one
        index.Upsert(new Document { Id = "doc-2", Body = "Alpha delta epsilon." });
        index.Delete("doc-1");
        index.Commit();

        var r2 = index.Search(new HybridQuery { Text = "alpha", TopK = 10 });
        Assert.Single(r2.Results);
        Assert.Equal("doc-2", r2.Results[0].Id);
    }

    [Fact]
    public void Upsert_WithEmbedding_VectorSearchWorks()
    {
        var embedding = new float[] { 0.6f, 0.8f };

        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "doc-1",
            Body = "Some text content.",
            Embedding = embedding
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Vector = embedding,
            TopK = 5
        });

        Assert.Single(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
    }

    [Fact]
    public void Delete_RemovesFromVectorIndex()
    {
        var embedding = new float[] { 0.6f, 0.8f };

        using var index = new MutableHybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "Some text content.",
                Embedding = embedding
            })
            .Build();

        // Verify vector search works
        var before = index.Search(new HybridQuery { Vector = embedding, TopK = 5 });
        Assert.Single(before.Results);

        index.Delete("doc-1");
        index.Commit();

        var after = index.Search(new HybridQuery { Vector = embedding, TopK = 5 });
        Assert.Empty(after.Results);
    }

    [Fact]
    public void Builder_SeedsInitialDocuments()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();

        using var index = new MutableHybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        Assert.Equal(5, index.GetStats().DocumentCount);

        var response = index.Search(new HybridQuery { Text = "neural network", TopK = 5 });
        Assert.NotNull(response);
    }

    [Fact]
    public void Dispose_ThenUpsert_Throws()
    {
        var index = new MutableHybridSearchIndexBuilder().Build();
        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            index.Upsert(new Document { Id = "doc-1", Body = "test" }));
    }

    [Fact]
    public void Dispose_ThenSearch_Throws()
    {
        var index = new MutableHybridSearchIndexBuilder().Build();
        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            index.Search(new HybridQuery { Text = "test", TopK = 5 }));
    }

    [Fact]
    public void Dispose_ThenCommit_Throws()
    {
        var index = new MutableHybridSearchIndexBuilder().Build();
        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => index.Commit());
    }

    [Fact]
    public void Dispose_ThenDelete_Throws()
    {
        var index = new MutableHybridSearchIndexBuilder().Build();
        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => index.Delete("doc-1"));
    }

    [Fact]
    public void HybridSearch_UpsertedDocuments_CombinesLexicalAndVector()
    {
        var embedding = new float[] { 0.6f, 0.8f };

        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "doc-1",
            Body = "Neural networks learn complex patterns from training data.",
            Embedding = embedding
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "neural network",
            Vector = embedding,
            TopK = 5
        });

        Assert.Single(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
    }

    [Fact]
    public async Task SearchAsync_Works()
    {
        using var index = new MutableHybridSearchIndexBuilder()
            .AddDocument(new Document { Id = "doc-1", Body = "Neural networks and deep learning." })
            .Build();

        var response = await index.SearchAsync(new HybridQuery
        {
            Text = "neural network",
            TopK = 5
        });

        Assert.NotNull(response);
        Assert.Single(response.Results);
    }

    [Fact]
    public void Upsert_Null_Throws()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        Assert.Throws<ArgumentNullException>(() => index.Upsert(null!));
    }

    [Fact]
    public void Delete_Null_Throws()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        Assert.Throws<ArgumentNullException>(() => index.Delete(null!));
    }
}
