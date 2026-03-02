using HybridSearch.Models;

namespace HybridSearch.Tests.Models;

public class IndexStatsTests
{
    [Fact]
    public void IndexStats_BasicProperties()
    {
        var stats = new IndexStats
        {
            DocumentCount = 100,
            EmbeddingDimension = 768,
            IndexBuildTimeMs = 450.5
        };

        Assert.Equal(100, stats.DocumentCount);
        Assert.Equal(768, stats.EmbeddingDimension);
        Assert.Equal(450.5, stats.IndexBuildTimeMs);
    }

    [Fact]
    public void IndexStats_NullEmbeddingDimension_WhenNoEmbeddings()
    {
        var stats = new IndexStats
        {
            DocumentCount = 50,
            EmbeddingDimension = null,
            IndexBuildTimeMs = 100
        };

        Assert.Null(stats.EmbeddingDimension);
    }
}
