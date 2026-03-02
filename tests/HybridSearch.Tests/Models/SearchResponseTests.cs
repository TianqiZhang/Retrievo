using HybridSearch.Models;

namespace HybridSearch.Tests.Models;

public class SearchResponseTests
{
    [Fact]
    public void SearchResponse_BasicProperties()
    {
        var response = new SearchResponse
        {
            Results = new[]
            {
                new SearchResult { Id = "doc-1", Score = 0.9 },
                new SearchResult { Id = "doc-2", Score = 0.7 }
            },
            QueryTimeMs = 12.5
        };

        Assert.Equal(2, response.Results.Count);
        Assert.Equal(12.5, response.QueryTimeMs);
    }

    [Fact]
    public void SearchResponse_EmptyResults()
    {
        var response = new SearchResponse
        {
            Results = Array.Empty<SearchResult>(),
            QueryTimeMs = 0.1
        };

        Assert.Empty(response.Results);
    }
}
