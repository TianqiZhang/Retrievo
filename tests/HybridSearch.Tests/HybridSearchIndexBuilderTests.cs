using HybridSearch.Abstractions;
using HybridSearch.Models;
using HybridSearch.Tests.TestData;
using NSubstitute;

namespace HybridSearch.Tests;

public class HybridSearchIndexBuilderTests
{
    [Fact]
    public void Build_WithNoDocuments_Throws()
    {
        var builder = new HybridSearchIndexBuilder();

        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }

    [Fact]
    public void AddDocument_Single()
    {
        var doc = new Document
        {
            Id = "doc-1",
            Body = "hello world",
            Embedding = new float[] { 1f, 0f, 0f }
        };

        var index = new HybridSearchIndexBuilder()
            .AddDocument(doc)
            .Build();

        using (index)
        {
            Assert.Equal(1, index.GetStats().DocumentCount);
        }
    }

    [Fact]
    public void AddDocuments_Collection()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();

        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            Assert.Equal(5, index.GetStats().DocumentCount);
        }
    }

    [Fact]
    public void AddFolder_LoadsTextFiles()
    {
        // Create a temp directory with test files
        var tempDir = Path.Combine(Path.GetTempPath(), $"hybridsearch_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            File.WriteAllText(Path.Combine(tempDir, "doc1.md"), "# Machine Learning\nNeural networks learn patterns.");
            File.WriteAllText(Path.Combine(tempDir, "doc2.txt"), "Kubernetes orchestrates containers.");
            File.WriteAllText(Path.Combine(tempDir, "ignored.json"), "{\"not\": \"indexed\"}");

            var index = new HybridSearchIndexBuilder()
                .AddFolder(tempDir)
                .Build();

            using (index)
            {
                // Should have loaded 2 files (md + txt), ignoring json
                Assert.Equal(2, index.GetStats().DocumentCount);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void AddFolder_NonexistentPath_Throws()
    {
        var builder = new HybridSearchIndexBuilder();

        Assert.Throws<DirectoryNotFoundException>(() =>
            builder.AddFolder(@"C:\nonexistent_path_xyz_12345"));
    }

    [Fact]
    public void AddFolder_SkipsEmptyFiles()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"hybridsearch_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            File.WriteAllText(Path.Combine(tempDir, "has_content.md"), "Some actual content here.");
            File.WriteAllText(Path.Combine(tempDir, "empty.md"), "");
            File.WriteAllText(Path.Combine(tempDir, "whitespace.txt"), "   \n  ");

            var index = new HybridSearchIndexBuilder()
                .AddFolder(tempDir)
                .Build();

            using (index)
            {
                Assert.Equal(1, index.GetStats().DocumentCount);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void AddFolder_Recursive_FindsSubdirectoryFiles()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"hybridsearch_test_{Guid.NewGuid():N}");
        var subDir = Path.Combine(tempDir, "subdir");
        Directory.CreateDirectory(subDir);

        try
        {
            File.WriteAllText(Path.Combine(tempDir, "root.md"), "Root document content.");
            File.WriteAllText(Path.Combine(subDir, "nested.md"), "Nested document content.");

            var index = new HybridSearchIndexBuilder()
                .AddFolder(tempDir, recursive: true)
                .Build();

            using (index)
            {
                Assert.Equal(2, index.GetStats().DocumentCount);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task BuildAsync_WithEmbeddingProvider_EmbedsDocumentsWithoutEmbeddings()
    {
        var docsWithoutEmbeddings = new List<Document>
        {
            new() { Id = "doc-1", Body = "machine learning algorithms" },
            new() { Id = "doc-2", Body = "kubernetes container orchestration" }
        };

        var mockProvider = Substitute.For<IEmbeddingProvider>();
        mockProvider.Dimensions.Returns(3);
        mockProvider.EmbedBatchAsync(Arg.Any<IReadOnlyList<string>>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(new[]
            {
                new float[] { 1f, 0f, 0f },
                new float[] { 0f, 1f, 0f }
            }));

        var index = await new HybridSearchIndexBuilder()
            .AddDocuments(docsWithoutEmbeddings)
            .WithEmbeddingProvider(mockProvider)
            .BuildAsync();

        using (index)
        {
            Assert.Equal(2, index.GetStats().DocumentCount);
            Assert.Equal(3, index.GetStats().EmbeddingDimension);

            // Verify batch embed was called
            await mockProvider.Received(1).EmbedBatchAsync(
                Arg.Is<IReadOnlyList<string>>(texts => texts.Count == 2),
                Arg.Any<CancellationToken>());
        }
    }

    [Fact]
    public async Task BuildAsync_DoesNotReEmbedDocumentsWithEmbeddings()
    {
        var docs = new List<Document>
        {
            new() { Id = "doc-1", Body = "already has embedding", Embedding = new float[] { 1f, 0f, 0f } },
            new() { Id = "doc-2", Body = "needs embedding" }
        };

        var mockProvider = Substitute.For<IEmbeddingProvider>();
        mockProvider.Dimensions.Returns(3);
        mockProvider.EmbedBatchAsync(Arg.Any<IReadOnlyList<string>>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(new[] { new float[] { 0f, 1f, 0f } }));

        var index = await new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .WithEmbeddingProvider(mockProvider)
            .BuildAsync();

        using (index)
        {
            // Only doc-2 should have been embedded
            await mockProvider.Received(1).EmbedBatchAsync(
                Arg.Is<IReadOnlyList<string>>(texts => texts.Count == 1 && texts[0] == "needs embedding"),
                Arg.Any<CancellationToken>());
        }
    }

    [Fact]
    public void FluentAPI_ChainsCorrectly()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();
        var mockFuser = Substitute.For<IFuser>();
        mockFuser.Fuse(Arg.Any<IReadOnlyList<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)>>(),
                Arg.Any<int>(), Arg.Any<int>(), Arg.Any<bool>())
            .Returns(Array.Empty<SearchResult>());

        // Verify fluent chaining compiles and works
        var builder = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .WithFuser(mockFuser);

        var index = builder.Build();
        using (index)
        {
            Assert.Equal(5, index.GetStats().DocumentCount);
        }
    }

    [Fact]
    public void DocumentsWithoutEmbeddings_StillIndexedLexically()
    {
        var docs = new List<Document>
        {
            new() { Id = "doc-1", Body = "machine learning neural networks" },
            new() { Id = "doc-2", Body = "kubernetes container deployment" }
        };

        var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        using (index)
        {
            Assert.Equal(2, index.GetStats().DocumentCount);
            Assert.Null(index.GetStats().EmbeddingDimension); // No embeddings

            // Lexical search should still work
            var response = index.Search(new HybridQuery
            {
                Text = "machine learning",
                TopK = 5
            });

            Assert.Single(response.Results);
            Assert.Equal("doc-1", response.Results[0].Id);
        }
    }

    /// <summary>
    /// Spec §5.2: "Markdown ingestion sanity: markdown file content is searchable
    /// (at least plain text extraction; no requirement for AST parsing)."
    /// </summary>
    [Fact]
    public void AddFolder_MarkdownContent_IsSearchable()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"hybridsearch_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            File.WriteAllText(Path.Combine(tempDir, "neural.md"),
                "# Neural Networks\nNeural networks learn complex patterns from training data. " +
                "Deep learning has transformed artificial intelligence research.");

            File.WriteAllText(Path.Combine(tempDir, "databases.txt"),
                "SQL queries retrieve structured data from relational tables. " +
                "Indexing accelerates query performance on large datasets.");

            using var index = new HybridSearchIndexBuilder()
                .AddFolder(tempDir)
                .Build();

            // Search for content from the markdown file
            var response = index.Search(new HybridQuery
            {
                Text = "neural networks deep learning",
                TopK = 5
            });

            Assert.True(response.Results.Count > 0,
                "Markdown file content should be searchable");
            Assert.Contains(response.Results, r => r.Id.Contains("neural.md"));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}
