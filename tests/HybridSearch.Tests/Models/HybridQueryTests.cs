using HybridSearch.Models;

namespace HybridSearch.Tests.Models;

public class HybridQueryTests
{
    [Fact]
    public void HybridQuery_DefaultValues()
    {
        var query = new HybridQuery();

        Assert.Null(query.Text);
        Assert.Null(query.Vector);
        Assert.Equal(10, query.TopK);
        Assert.Equal(50, query.LexicalK);
        Assert.Equal(50, query.VectorK);
        Assert.Equal(1f, query.LexicalWeight);
        Assert.Equal(1f, query.VectorWeight);
        Assert.Equal(60, query.RrfK);
        Assert.False(query.Explain);
        Assert.Null(query.MetadataFilters);
    }

    [Fact]
    public void HybridQuery_WithText()
    {
        var query = new HybridQuery { Text = "machine learning" };

        Assert.Equal("machine learning", query.Text);
        Assert.Null(query.Vector);
    }

    [Fact]
    public void HybridQuery_WithVector()
    {
        var vec = new float[] { 0.1f, 0.2f, 0.3f };
        var query = new HybridQuery { Vector = vec };

        Assert.Null(query.Text);
        Assert.Same(vec, query.Vector);
    }

    [Fact]
    public void HybridQuery_WithCustomWeights()
    {
        var query = new HybridQuery
        {
            Text = "test",
            Vector = new float[] { 1f },
            LexicalWeight = 0.3f,
            VectorWeight = 0.7f,
            RrfK = 30,
            TopK = 5
        };

        Assert.Equal(0.3f, query.LexicalWeight);
        Assert.Equal(0.7f, query.VectorWeight);
        Assert.Equal(30, query.RrfK);
        Assert.Equal(5, query.TopK);
    }

    [Fact]
    public void HybridQuery_RecordEquality()
    {
        var q1 = new HybridQuery { Text = "hello", TopK = 5 };
        var q2 = new HybridQuery { Text = "hello", TopK = 5 };
        var q3 = new HybridQuery { Text = "hello", TopK = 10 };

        Assert.Equal(q1, q2);
        Assert.NotEqual(q1, q3);
    }

    [Fact]
    public void HybridQuery_WithExplain()
    {
        var query = new HybridQuery { Text = "test", Explain = true };

        Assert.True(query.Explain);
    }
}
