using HybridSearch.Models;

namespace HybridSearch.Tests.Models;

public class DocumentTests
{
    [Fact]
    public void Document_RequiresIdAndBody()
    {
        var doc = new Document { Id = "doc-1", Body = "Hello world" };

        Assert.Equal("doc-1", doc.Id);
        Assert.Equal("Hello world", doc.Body);
        Assert.Null(doc.Title);
        Assert.Null(doc.Embedding);
        Assert.Null(doc.Metadata);
    }

    [Fact]
    public void Document_WithAllFields()
    {
        var embedding = new float[] { 0.1f, 0.2f, 0.3f };
        var metadata = new Dictionary<string, string> { ["sourcePath"] = "/docs/test.md" };

        var doc = new Document
        {
            Id = "doc-2",
            Title = "Test Document",
            Body = "Content here",
            Embedding = embedding,
            Metadata = metadata
        };

        Assert.Equal("doc-2", doc.Id);
        Assert.Equal("Test Document", doc.Title);
        Assert.Equal("Content here", doc.Body);
        Assert.Same(embedding, doc.Embedding);
        Assert.Equal("/docs/test.md", doc.Metadata!["sourcePath"]);
    }

    [Fact]
    public void Document_NullEmbedding_IsValid()
    {
        // A document with no embedding is valid — it participates in lexical search only.
        var doc = new Document { Id = "doc-no-vec", Body = "Lexical only document" };

        Assert.Null(doc.Embedding);
    }
}
