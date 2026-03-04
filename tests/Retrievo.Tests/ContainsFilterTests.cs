using Retrievo.Models;

namespace Retrievo.Tests;

public class ContainsFilterTests
{
    private static MutableHybridSearchIndex BuildMutableIndexWithMultiValueMetadata()
    {
        var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "event-1",
            Body = "Deployed new microservice to production cluster.",
            Metadata = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-alpha|proj-beta",
                ["service"] = "deploy",
                ["tags"] = "infra|production|k8s"
            }
        });
        index.Upsert(new Document
        {
            Id = "event-2",
            Body = "Fixed authentication bug in login service.",
            Metadata = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-beta|proj-gamma",
                ["service"] = "auth",
                ["tags"] = "bugfix|security"
            }
        });
        index.Upsert(new Document
        {
            Id = "event-3",
            Body = "Updated documentation for API endpoints.",
            Metadata = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-alpha",
                ["service"] = "docs",
                ["tags"] = "documentation"
            }
        });
        index.Upsert(new Document
        {
            Id = "event-4",
            Body = "Deployed hotfix to authentication service.",
            Metadata = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-gamma",
                ["service"] = "deploy",
                ["tags"] = "infra|hotfix|security"
            }
        });
        index.Commit();

        return index;
    }

    [Fact]
    public void ContainsFilter_SingleValue_MatchesDocWithThatValue()
    {
        using var index = BuildMutableIndexWithMultiValueMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "deployed authentication service",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-alpha"
            }
        });

        // event-1 has proj-alpha|proj-beta, event-3 has proj-alpha
        Assert.All(response.Results, r =>
            Assert.True(r.Id == "event-1" || r.Id == "event-3",
                $"Expected event-1 or event-3 but got {r.Id}"));
        Assert.True(response.Results.Count >= 1);
    }

    [Fact]
    public void ContainsFilter_MatchesMultipleDocuments()
    {
        using var index = BuildMutableIndexWithMultiValueMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "deployed authentication fixed documentation",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-beta"
            }
        });

        // event-1 has proj-alpha|proj-beta, event-2 has proj-beta|proj-gamma
        Assert.All(response.Results, r =>
            Assert.True(r.Id == "event-1" || r.Id == "event-2",
                $"Expected event-1 or event-2 but got {r.Id}"));
        Assert.True(response.Results.Count >= 1);
    }

    [Fact]
    public void ContainsFilter_NoMatch_ReturnsEmpty()
    {
        using var index = BuildMutableIndexWithMultiValueMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "deployed authentication",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-nonexistent"
            }
        });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void ContainsFilter_PartialValueDoesNotMatch()
    {
        using var index = BuildMutableIndexWithMultiValueMetadata();

        // "proj-alph" is a substring of "proj-alpha" but not a delimited element
        var response = index.Search(new HybridQuery
        {
            Text = "deployed documentation service",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-alph"
            }
        });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void ContainsFilter_SingleElementValue_Matches()
    {
        using var index = BuildMutableIndexWithMultiValueMetadata();

        // event-3 has project_ids = "proj-alpha" (single value, no delimiter)
        var response = index.Search(new HybridQuery
        {
            Text = "documentation API endpoints",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["project_ids"] = "proj-alpha"
            }
        });

        Assert.Contains(response.Results, r => r.Id == "event-3");
    }

    [Fact]
    public void ContainsFilter_CombinedWithExactFilter()
    {
        using var index = BuildMutableIndexWithMultiValueMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "deployed authentication service hotfix",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["service"] = "deploy" },
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["tags"] = "security"
            }
        });

        // Only event-4 is service=deploy AND tags contains "security"
        Assert.Single(response.Results);
        Assert.Equal("event-4", response.Results[0].Id);
    }

    [Fact]
    public void ContainsFilter_CombinedWithRangeFilter()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "e1",
            Body = "Early event with multiple projects.",
            Metadata = new Dictionary<string, string>
            {
                ["timestamp"] = "2025-01-10T00:00:00Z",
                ["project_ids"] = "p1|p2"
            }
        });
        index.Upsert(new Document
        {
            Id = "e2",
            Body = "Late event with multiple projects.",
            Metadata = new Dictionary<string, string>
            {
                ["timestamp"] = "2025-06-15T00:00:00Z",
                ["project_ids"] = "p1|p3"
            }
        });
        index.Upsert(new Document
        {
            Id = "e3",
            Body = "Late event with different projects.",
            Metadata = new Dictionary<string, string>
            {
                ["timestamp"] = "2025-06-20T00:00:00Z",
                ["project_ids"] = "p2|p3"
            }
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "event projects",
            TopK = 10,
            MetadataRangeFilters = new[]
            {
                new MetadataRangeFilter { Key = "timestamp", Min = "2025-06-01T00:00:00Z" }
            },
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["project_ids"] = "p1"
            }
        });

        // Only e2 has timestamp >= June AND project_ids contains p1
        Assert.Single(response.Results);
        Assert.Equal("e2", response.Results[0].Id);
    }

    [Fact]
    public void ContainsFilter_MissingKey_FiltersOut()
    {
        using var index = BuildMutableIndexWithMultiValueMetadata();

        var response = index.Search(new HybridQuery
        {
            Text = "deployed authentication",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["nonexistent_key"] = "value"
            }
        });

        Assert.Empty(response.Results);
    }

    [Fact]
    public void ContainsFilter_NullMetadataDocument_FilteredOut()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "with-meta",
            Body = "Document with project metadata.",
            Metadata = new Dictionary<string, string>
            {
                ["project_ids"] = "p1|p2"
            }
        });
        index.Upsert(new Document
        {
            Id = "no-meta",
            Body = "Document with no metadata.",
            Metadata = null
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "document project metadata",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["project_ids"] = "p1"
            }
        });

        Assert.Single(response.Results);
        Assert.Equal("with-meta", response.Results[0].Id);
    }

    [Fact]
    public void ContainsFilter_CustomDelimiter()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "doc-1",
            Body = "Document with comma-separated tags.",
            Metadata = new Dictionary<string, string>
            {
                ["tags"] = "alpha,beta,gamma"
            }
        });
        index.Upsert(new Document
        {
            Id = "doc-2",
            Body = "Document with different tags.",
            Metadata = new Dictionary<string, string>
            {
                ["tags"] = "delta,epsilon"
            }
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "document tags",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string> { ["tags"] = "beta" },
            MetadataContainsDelimiter = ','
        });

        Assert.Single(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
    }

    [Fact]
    public void ContainsFilter_WorksWithImmutableIndex()
    {
        using var index = new HybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "First document with multiple projects.",
                Metadata = new Dictionary<string, string>
                {
                    ["project_ids"] = "p1|p2"
                }
            })
            .AddDocument(new Document
            {
                Id = "doc-2",
                Body = "Second document with different projects.",
                Metadata = new Dictionary<string, string>
                {
                    ["project_ids"] = "p3|p4"
                }
            })
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "document projects",
            TopK = 10,
            MetadataContainsFilters = new Dictionary<string, string>
            {
                ["project_ids"] = "p2"
            }
        });

        Assert.Single(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
    }
}
