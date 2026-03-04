using Retrievo.Models;

namespace Retrievo.Tests;

public class RangeFilterTests
{
    private static MutableHybridSearchIndex BuildMutableIndexWithTimestamps()
    {
        var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "event-1",
            Body = "User logged in from mobile device.",
            Metadata = new Dictionary<string, string>
            {
                ["timestamp"] = "2025-01-15T10:00:00Z",
                ["service"] = "auth",
                ["source"] = "mobile"
            }
        });
        index.Upsert(new Document
        {
            Id = "event-2",
            Body = "User updated profile settings.",
            Metadata = new Dictionary<string, string>
            {
                ["timestamp"] = "2025-02-20T14:30:00Z",
                ["service"] = "profile",
                ["source"] = "web"
            }
        });
        index.Upsert(new Document
        {
            Id = "event-3",
            Body = "User logged in from web browser.",
            Metadata = new Dictionary<string, string>
            {
                ["timestamp"] = "2025-03-10T09:15:00Z",
                ["service"] = "auth",
                ["source"] = "web"
            }
        });
        index.Upsert(new Document
        {
            Id = "event-4",
            Body = "User logged out of the system.",
            Metadata = new Dictionary<string, string>
            {
                ["timestamp"] = "2025-04-05T16:45:00Z",
                ["service"] = "auth",
                ["source"] = "mobile"
            }
        });
        index.Commit();

        return index;
    }

    private static HybridSearchIndex BuildImmutableIndexWithTimestamps()
    {
        return new HybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "event-1",
                Body = "User logged in from mobile device.",
                Metadata = new Dictionary<string, string>
                {
                    ["timestamp"] = "2025-01-15T10:00:00Z",
                    ["service"] = "auth",
                    ["source"] = "mobile"
                }
            })
            .AddDocument(new Document
            {
                Id = "event-2",
                Body = "User updated profile settings.",
                Metadata = new Dictionary<string, string>
                {
                    ["timestamp"] = "2025-02-20T14:30:00Z",
                    ["service"] = "profile",
                    ["source"] = "web"
                }
            })
            .AddDocument(new Document
            {
                Id = "event-3",
                Body = "User logged in from web browser.",
                Metadata = new Dictionary<string, string>
                {
                    ["timestamp"] = "2025-03-10T09:15:00Z",
                    ["service"] = "auth",
                    ["source"] = "web"
                }
            })
            .AddDocument(new Document
            {
                Id = "event-4",
                Body = "User logged out of the system.",
                Metadata = new Dictionary<string, string>
                {
                    ["timestamp"] = "2025-04-05T16:45:00Z",
                    ["service"] = "auth",
                    ["source"] = "mobile"
                }
            })
            .Build();
    }

    [Fact]
    public void RangeFilter_MinOnly_ReturnsDocsAfterMin()
    {
        using var index = BuildMutableIndexWithTimestamps();

        var response = index.Search(new HybridQuery
        {
            Text = "user logged",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter { Key = "timestamp", Min = "2025-03-01T00:00:00Z" }
            }
        });

        // event-3 (March) and event-4 (April) should match
        Assert.Equal(2, response.Results.Count);
        Assert.All(response.Results, r =>
            Assert.True(r.Id == "event-3" || r.Id == "event-4",
                $"Expected event-3 or event-4 but got {r.Id}"));
    }

    [Fact]
    public void RangeFilter_MaxOnly_ReturnsDocsBeforeMax()
    {
        using var index = BuildMutableIndexWithTimestamps();

        var response = index.Search(new HybridQuery
        {
            Text = "user logged updated",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter { Key = "timestamp", Max = "2025-02-28T23:59:59Z" }
            }
        });

        // event-1 (January) and event-2 (February) should match
        Assert.Equal(2, response.Results.Count);
        Assert.All(response.Results, r =>
            Assert.True(r.Id == "event-1" || r.Id == "event-2",
                $"Expected event-1 or event-2 but got {r.Id}"));
    }

    [Fact]
    public void RangeFilter_MinAndMax_ReturnsBetween()
    {
        using var index = BuildMutableIndexWithTimestamps();

        var response = index.Search(new HybridQuery
        {
            Text = "user logged updated profile",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter
                {
                    Key = "timestamp",
                    Min = "2025-02-01T00:00:00Z",
                    Max = "2025-03-31T23:59:59Z"
                }
            }
        });

        // event-2 (Feb) and event-3 (March) should match
        Assert.Equal(2, response.Results.Count);
        Assert.All(response.Results, r =>
            Assert.True(r.Id == "event-2" || r.Id == "event-3",
                $"Expected event-2 or event-3 but got {r.Id}"));
    }

    [Fact]
    public void RangeFilter_NoMatch_ReturnsEmpty()
    {
        using var index = BuildMutableIndexWithTimestamps();

        var response = index.Search(new HybridQuery
        {
            Text = "user logged",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter
                {
                    Key = "timestamp",
                    Min = "2026-01-01T00:00:00Z",
                    Max = "2026-12-31T23:59:59Z"
                }
            }
        });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void RangeFilter_CombinedWithExactFilter()
    {
        using var index = BuildMutableIndexWithTimestamps();

        var response = index.Search(new HybridQuery
        {
            Text = "user logged",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["service"] = "auth" },
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter { Key = "timestamp", Min = "2025-03-01T00:00:00Z" }
            }
        });

        // Only auth events after March: event-3 and event-4
        Assert.Equal(2, response.Results.Count);
        Assert.All(response.Results, r =>
            Assert.True(r.Id == "event-3" || r.Id == "event-4",
                $"Expected event-3 or event-4 but got {r.Id}"));
    }

    [Fact]
    public void RangeFilter_MissingKey_FiltersOut()
    {
        using var index = BuildMutableIndexWithTimestamps();

        var response = index.Search(new HybridQuery
        {
            Text = "user logged",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter { Key = "nonexistent", Min = "anything" }
            }
        });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void RangeFilter_InclusiveBounds()
    {
        using var index = BuildMutableIndexWithTimestamps();

        // Use exact timestamp as Min — should include that document (inclusive)
        var response = index.Search(new HybridQuery
        {
            Text = "user logged updated profile",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter
                {
                    Key = "timestamp",
                    Min = "2025-02-20T14:30:00Z",
                    Max = "2025-02-20T14:30:00Z"
                }
            }
        });

        Assert.Single(response.Results);
        Assert.Equal("event-2", response.Results[0].Id);
    }

    [Fact]
    public void RangeFilter_WorksWithImmutableIndex()
    {
        using var index = BuildImmutableIndexWithTimestamps();

        var response = index.Search(new HybridQuery
        {
            Text = "user logged",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter { Key = "timestamp", Min = "2025-03-01T00:00:00Z" }
            }
        });

        Assert.Equal(2, response.Results.Count);
        Assert.All(response.Results, r =>
            Assert.True(r.Id == "event-3" || r.Id == "event-4",
                $"Expected event-3 or event-4 but got {r.Id}"));
    }

    [Fact]
    public void RangeFilter_NullMetadataDocument_FilteredOut()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "with-meta",
            Body = "Document with metadata and timestamp.",
            Metadata = new Dictionary<string, string>
            {
                ["timestamp"] = "2025-06-01T00:00:00Z"
            }
        });
        index.Upsert(new Document
        {
            Id = "no-meta",
            Body = "Document with no metadata at all.",
            Metadata = null
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "document metadata timestamp",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter { Key = "timestamp", Min = "2025-01-01T00:00:00Z" }
            }
        });

        Assert.Single(response.Results);
        Assert.Equal("with-meta", response.Results[0].Id);
    }

    [Fact]
    public void MetadataRangeFilter_Validate_ThrowsWhenBothBoundsNull()
    {
        var filter = new MetadataRangeFilter { Key = "timestamp" };

        Assert.Throws<ArgumentException>(() => filter.Validate());
    }

    [Fact]
    public void MetadataRangeFilter_Validate_SucceedsWithMinOnly()
    {
        var filter = new MetadataRangeFilter { Key = "timestamp", Min = "2025-01-01" };

        // Should not throw
        filter.Validate();
    }

    [Fact]
    public void MetadataRangeFilter_Validate_SucceedsWithMaxOnly()
    {
        var filter = new MetadataRangeFilter { Key = "timestamp", Max = "2025-12-31" };

        // Should not throw
        filter.Validate();
    }

    [Fact]
    public void HybridQuery_ValidateBoosts_ThrowsForInvalidRangeFilter()
    {
        var query = new HybridQuery
        {
            Text = "test",
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter { Key = "timestamp" } // Both Min and Max null
            }
        };

        Assert.Throws<ArgumentException>(() => query.ValidateBoosts());
    }
}
