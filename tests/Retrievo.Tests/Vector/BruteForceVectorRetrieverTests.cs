using Retrievo.Vector;

namespace Retrievo.Tests.Vector;

public class BruteForceVectorRetrieverTests
{
    [Fact]
    public void IdenticalVectors_SimilarityOne()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });

        var results = retriever.Search(new float[] { 1f, 0f, 0f }, topK: 10);

        Assert.Single(results);
        Assert.Equal("doc-1", results[0].Id);
        Assert.Equal(1.0, results[0].Score, tolerance: 1e-5);
    }

    [Fact]
    public void OrthogonalVectors_SimilarityZero()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });

        var results = retriever.Search(new float[] { 0f, 1f, 0f }, topK: 10);

        Assert.Single(results);
        Assert.Equal(0.0, results[0].Score, tolerance: 1e-5);
    }

    [Fact]
    public void TopK_CorrectOrdering()
    {
        var retriever = new BruteForceVectorRetriever();
        // doc-a: pointing along x axis
        retriever.Add("doc-a", new float[] { 1f, 0f, 0f });
        // doc-b: pointing mostly along x with some y
        retriever.Add("doc-b", new float[] { 0.9f, 0.4f, 0f });
        // doc-c: pointing along y axis
        retriever.Add("doc-c", new float[] { 0f, 1f, 0f });
        // doc-d: pointing somewhat toward x
        retriever.Add("doc-d", new float[] { 0.5f, 0.5f, 0.7f });

        // Query along x axis
        var results = retriever.Search(new float[] { 1f, 0f, 0f }, topK: 4);

        Assert.Equal(4, results.Count);
        // doc-a should be first (perfect match), doc-c should be last (orthogonal)
        Assert.Equal("doc-a", results[0].Id);
        Assert.Equal("doc-c", results[3].Id);

        // Verify descending score order
        for (int i = 1; i < results.Count; i++)
            Assert.True(results[i - 1].Score >= results[i].Score);

        // Verify ranks are 1-based
        for (int i = 0; i < results.Count; i++)
            Assert.Equal(i + 1, results[i].Rank);
    }

    [Fact]
    public void TopK_LargerThanCorpus_ReturnsAll()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f });
        retriever.Add("doc-2", new float[] { 0f, 1f });

        var results = retriever.Search(new float[] { 1f, 0f }, topK: 100);

        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void EmptyCorpus_ReturnsEmpty()
    {
        var retriever = new BruteForceVectorRetriever();
        var results = retriever.Search(new float[] { 1f, 0f, 0f }, topK: 10);

        Assert.Empty(results);
    }

    [Fact]
    public void DimensionMismatch_OnAdd_Throws()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });

        Assert.Throws<ArgumentException>(() =>
            retriever.Add("doc-2", new float[] { 1f, 0f }));
    }

    [Fact]
    public void DimensionMismatch_OnSearch_Throws()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });

        Assert.Throws<ArgumentException>(() =>
            retriever.Search(new float[] { 1f, 0f }, topK: 10));
    }

    [Fact]
    public void PreNormalization_DoesNotAffectResults()
    {
        var retriever = new BruteForceVectorRetriever();
        // Add with non-unit vectors — should be normalized internally
        retriever.Add("doc-1", new float[] { 100f, 0f, 0f });
        retriever.Add("doc-2", new float[] { 0f, 0.001f, 0f });

        // Search with non-unit query
        var results = retriever.Search(new float[] { 50f, 0f, 0f }, topK: 2);

        Assert.Equal("doc-1", results[0].Id);
        Assert.Equal(1.0, results[0].Score, tolerance: 1e-5);
    }

    [Fact]
    public void Dimensions_PropertySetOnFirstAdd()
    {
        var retriever = new BruteForceVectorRetriever();
        Assert.Equal(0, retriever.Dimensions);

        retriever.Add("doc-1", new float[] { 1f, 2f, 3f });
        Assert.Equal(3, retriever.Dimensions);
    }

    [Fact]
    public void Count_ReflectsAddedDocuments()
    {
        var retriever = new BruteForceVectorRetriever();
        Assert.Equal(0, retriever.Count);

        retriever.Add("doc-1", new float[] { 1f, 0f });
        Assert.Equal(1, retriever.Count);

        retriever.Add("doc-2", new float[] { 0f, 1f });
        Assert.Equal(2, retriever.Count);
    }

    [Fact]
    public void Add_NaNEmbedding_Throws()
    {
        var retriever = new BruteForceVectorRetriever();

        Assert.Throws<ArgumentException>(() =>
            retriever.Add("doc-1", new float[] { 1f, float.NaN, 0f }));
    }

    [Fact]
    public void Update_InfinityEmbedding_Throws()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });

        Assert.Throws<ArgumentException>(() =>
            retriever.Update("doc-1", new float[] { 1f, float.PositiveInfinity, 0f }));
    }

    [Fact]
    public void Search_NaNQueryVector_Throws()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });

        Assert.Throws<ArgumentException>(() =>
            retriever.Search(new float[] { float.NaN, 0f, 0f }, topK: 10));
    }

    [Fact]
    public void Add_InvalidFirstEmbedding_DoesNotSetDimensions()
    {
        var retriever = new BruteForceVectorRetriever();

        Assert.Throws<ArgumentException>(() =>
            retriever.Add("bad", new float[] { float.NaN, 1f, 2f }));

        Assert.Equal(0, retriever.Dimensions);

        retriever.Add("good", new float[] { 1f, 2f });
        Assert.Equal(2, retriever.Dimensions);
        Assert.Equal(1, retriever.Count);
    }

    [Fact]
    public void Update_InvalidFirstEmbedding_DoesNotSetDimensions()
    {
        var retriever = new BruteForceVectorRetriever();

        Assert.Throws<ArgumentException>(() =>
            retriever.Update("bad", new float[] { 1f, float.PositiveInfinity, 2f }));

        Assert.Equal(0, retriever.Dimensions);

        retriever.Update("good", new float[] { 1f, 2f, 3f, 4f });
        Assert.Equal(4, retriever.Dimensions);
        Assert.Equal(1, retriever.Count);
    }

    [Fact]
    public void Search_CancelledToken_ThrowsOperationCanceledException()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() =>
            retriever.Search(new float[] { 1f, 0f, 0f }, topK: 10, cts.Token));
    }

    [Fact]
    public void Search_WithCancellationToken_ReturnsNormalResults()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });
        retriever.Add("doc-2", new float[] { 0f, 1f, 0f });

        using var cts = new CancellationTokenSource();
        var results = retriever.Search(new float[] { 1f, 0f, 0f }, topK: 10, cts.Token);

        Assert.Equal(2, results.Count);
        Assert.Equal("doc-1", results[0].Id);
        Assert.Equal(1.0, results[0].Score, tolerance: 1e-5);
    }

    [Fact]
    public void Search_DefaultCancellationToken_ReturnsNormalResults()
    {
        var retriever = new BruteForceVectorRetriever();
        retriever.Add("doc-1", new float[] { 1f, 0f, 0f });

        var results = retriever.Search(new float[] { 1f, 0f, 0f }, topK: 10, CancellationToken.None);

        Assert.Single(results);
        Assert.Equal("doc-1", results[0].Id);
    }
}
