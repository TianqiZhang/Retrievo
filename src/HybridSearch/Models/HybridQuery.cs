namespace HybridSearch.Models;

/// <summary>
/// A hybrid search query supporting lexical, vector, or combined retrieval.
/// </summary>
public sealed record HybridQuery
{
    /// <summary>
    /// Text query for lexical (BM25) retrieval. Null to skip lexical search.
    /// </summary>
    public string? Text { get; init; }

    /// <summary>
    /// Vector query for similarity retrieval. Null to skip vector search.
    /// If <see cref="Text"/> is provided and this is null, the engine will attempt
    /// to embed the text automatically when an <see cref="Abstractions.IEmbeddingProvider"/> is available.
    /// </summary>
    public float[]? Vector { get; init; }

    /// <summary>
    /// Number of final results to return after fusion.
    /// </summary>
    public int TopK { get; init; } = 10;

    /// <summary>
    /// Number of candidates to retrieve from the lexical index before fusion.
    /// Should be >= <see cref="TopK"/> to give RRF enough candidates.
    /// </summary>
    public int LexicalK { get; init; } = 50;

    /// <summary>
    /// Number of candidates to retrieve from the vector index before fusion.
    /// Should be >= <see cref="TopK"/> to give RRF enough candidates.
    /// </summary>
    public int VectorK { get; init; } = 50;

    /// <summary>
    /// Weight applied to lexical scores during RRF fusion.
    /// </summary>
    public float LexicalWeight { get; init; } = 1f;

    /// <summary>
    /// Weight applied to vector scores during RRF fusion.
    /// </summary>
    public float VectorWeight { get; init; } = 1f;

    /// <summary>
    /// The k constant in the RRF formula: score = weight * 1/(RrfK + rank).
    /// Default is 60, from the original RRF paper (Cormack, Clarke &amp; Butt, 2009).
    /// </summary>
    public int RrfK { get; init; } = 60;

    /// <summary>
    /// When true, each result includes detailed score breakdown in <see cref="SearchResult.Explain"/>.
    /// </summary>
    public bool Explain { get; init; }

    /// <summary>
    /// Optional exact-match metadata filters. Only documents whose metadata contains
    /// all specified key-value pairs will be returned. Phase 2 feature; null means no filtering.
    /// </summary>
    public IReadOnlyDictionary<string, string>? MetadataFilters { get; init; }

    /// <summary>
    /// Boost multiplier for the title field during lexical search.
    /// Higher values increase the relevance of title matches. Default is 1.0.
    /// </summary>
    public float TitleBoost { get; init; } = 1f;

    /// <summary>
    /// Boost multiplier for the body field during lexical search.
    /// Higher values increase the relevance of body matches. Default is 1.0.
    /// </summary>
    public float BodyBoost { get; init; } = 1f;
}
