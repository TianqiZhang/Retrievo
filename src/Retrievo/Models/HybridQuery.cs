namespace Retrievo.Models;

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
    /// Weight applied to lexical scores during RRF fusion. Default is 0.5,
    /// tuned via BEIR benchmark sweeps across NFCorpus and SciFact datasets.
    /// </summary>
    public float LexicalWeight { get; init; } = 0.5f;

    /// <summary>
    /// Weight applied to vector scores during RRF fusion. Default is 1.0.
    /// </summary>
    public float VectorWeight { get; init; } = 1f;

    /// <summary>
    /// The k constant in the RRF formula: score = weight * 1/(RrfK + rank).
    /// Default is 20, tuned via BEIR benchmark sweeps across NFCorpus and SciFact datasets.
    /// </summary>
    public int RrfK { get; init; } = 20;

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
    /// Optional range filters for metadata values. Each filter specifies a key and inclusive Min/Max bounds.
    /// Values are compared using ordinal string comparison, which works correctly for ISO 8601 timestamps
    /// and zero-padded numeric strings. All range filters must match (AND semantics). Null means no range filtering.
    /// </summary>
    public IReadOnlyList<MetadataRangeFilter>? MetadataRangeFilters { get; init; }

    /// <summary>
    /// Optional contains-match metadata filters. For each key-value pair, the document's metadata value
    /// for that key is split by the <see cref="MetadataContainsDelimiter"/> and the filter passes if any
    /// element equals the filter value (ordinal comparison). Useful for multi-value metadata fields.
    /// All contains filters must match (AND semantics). Null means no contains filtering.
    /// </summary>
    public IReadOnlyDictionary<string, string>? MetadataContainsFilters { get; init; }

    /// <summary>
    /// Delimiter character used to split metadata values for <see cref="MetadataContainsFilters"/>.
    /// Default is '|'. For example, a metadata value "proj1|proj2|proj3" with delimiter '|'
    /// and filter value "proj2" would match.
    /// </summary>
    public char MetadataContainsDelimiter { get; init; } = '|';

    /// <summary>
    /// Boost multiplier for the title field during lexical search.
    /// Higher values increase the relevance of title matches. Default is 0.5,
    /// tuned via BEIR benchmark sweeps across NFCorpus and SciFact datasets.
    /// </summary>
    public float TitleBoost { get; init; } = 0.5f;

    /// <summary>
    /// Boost multiplier for the body field during lexical search.
    /// Higher values increase the relevance of body matches. Default is 1.0.
    /// </summary>
    public float BodyBoost { get; init; } = 1f;

    /// <summary>
    /// Validates that TitleBoost and BodyBoost are finite non-negative values,
    /// and that any range filters have at least one bound specified.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a boost value is NaN, infinite, or negative.</exception>
    /// <exception cref="ArgumentException">Thrown when a range filter has both Min and Max null.</exception>
    internal void ValidateBoosts()
    {
        if (float.IsNaN(TitleBoost) || float.IsInfinity(TitleBoost) || TitleBoost < 0)
            throw new ArgumentOutOfRangeException(nameof(TitleBoost), $"TitleBoost must be a finite non-negative value, got {TitleBoost}.");
        if (float.IsNaN(BodyBoost) || float.IsInfinity(BodyBoost) || BodyBoost < 0)
            throw new ArgumentOutOfRangeException(nameof(BodyBoost), $"BodyBoost must be a finite non-negative value, got {BodyBoost}.");

        if (MetadataRangeFilters is not null)
        {
            foreach (var filter in MetadataRangeFilters)
            {
                ArgumentNullException.ThrowIfNull(filter, nameof(MetadataRangeFilters));
                filter.Validate();
            }
        }
    }
}
