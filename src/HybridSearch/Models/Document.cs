namespace HybridSearch.Models;

/// <summary>
/// Represents a document in the search index.
/// </summary>
public sealed class Document
{
    /// <summary>
    /// Stable unique identifier for the document.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Optional document title (e.g., filename without extension).
    /// </summary>
    public string? Title { get; init; }

    /// <summary>
    /// Primary text content of the document.
    /// </summary>
    public required string Body { get; init; }

    /// <summary>
    /// Optional pre-computed embedding vector. If null and an <see cref="Abstractions.IEmbeddingProvider"/>
    /// is configured, the embedding will be generated automatically from <see cref="Body"/>.
    /// </summary>
    public float[]? Embedding { get; init; }

    /// <summary>
    /// Optional key-value metadata (e.g., sourcePath, lastModified, category).
    /// </summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }
}
