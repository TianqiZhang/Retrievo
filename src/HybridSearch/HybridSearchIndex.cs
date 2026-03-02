using System.Diagnostics;
using HybridSearch.Abstractions;
using HybridSearch.Fusion;
using HybridSearch.Lexical;
using HybridSearch.Models;
using HybridSearch.Vector;

namespace HybridSearch;

/// <summary>
/// Core hybrid search index that orchestrates lexical retrieval (BM25),
/// vector retrieval (cosine similarity), and RRF fusion.
/// Thread-safe for concurrent reads.
/// </summary>
public sealed class HybridSearchIndex : IHybridSearchIndex
{
    private readonly LuceneLexicalRetriever _lexicalRetriever;
    private readonly BruteForceVectorRetriever _vectorRetriever;
    private readonly IFuser _fuser;
    private readonly IEmbeddingProvider? _embeddingProvider;
    private readonly Dictionary<string, Document> _documents;
    private readonly IndexStats _stats;
    private bool _disposed;

    internal HybridSearchIndex(
        LuceneLexicalRetriever lexicalRetriever,
        BruteForceVectorRetriever vectorRetriever,
        IFuser fuser,
        IEmbeddingProvider? embeddingProvider,
        Dictionary<string, Document> documents,
        IndexStats stats)
    {
        _lexicalRetriever = lexicalRetriever;
        _vectorRetriever = vectorRetriever;
        _fuser = fuser;
        _embeddingProvider = embeddingProvider;
        _documents = documents;
        _stats = stats;
    }

    /// <inheritdoc/>
    public SearchResponse Search(HybridQuery query)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(query);
        ValidateBoosts(query);

        var totalSw = Stopwatch.StartNew();

        // If text is provided, no vector, and we have an embedding provider, embed synchronously
        float[]? queryVector = query.Vector;
        double? embeddingTimeMs = null;
        if (queryVector is null && query.Text is not null && _embeddingProvider is not null)
        {
            var embedSw = Stopwatch.StartNew();
            queryVector = _embeddingProvider.EmbedAsync(query.Text).GetAwaiter().GetResult();
            embedSw.Stop();
            embeddingTimeMs = embedSw.Elapsed.TotalMilliseconds;
        }

        return ExecuteSearch(query, queryVector, embeddingTimeMs, totalSw);
    }

    /// <inheritdoc/>
    public async Task<SearchResponse> SearchAsync(HybridQuery query, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(query);
        ValidateBoosts(query);

        var totalSw = Stopwatch.StartNew();

        // If text is provided, no vector, and we have an embedding provider, embed asynchronously
        float[]? queryVector = query.Vector;
        double? embeddingTimeMs = null;
        if (queryVector is null && query.Text is not null && _embeddingProvider is not null)
        {
            var embedSw = Stopwatch.StartNew();
            queryVector = await _embeddingProvider.EmbedAsync(query.Text, ct).ConfigureAwait(false);
            embedSw.Stop();
            embeddingTimeMs = embedSw.Elapsed.TotalMilliseconds;
        }

        return ExecuteSearch(query, queryVector, embeddingTimeMs, totalSw);
    }

    /// <inheritdoc/>
    public IndexStats GetStats()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _stats;
    }

    private SearchResponse ExecuteSearch(HybridQuery query, float[]? queryVector, double? embeddingTimeMs, Stopwatch totalSw)
    {
        double? lexicalTimeMs = null;
        double? vectorTimeMs = null;
        double fusionTimeMs = 0;
        double? filterTimeMs = null;

        bool hasMetadataFilters = query.MetadataFilters is not null && query.MetadataFilters.Count > 0;

        // Over-retrieve when metadata filters are applied to compensate for filtered-out results
        int overRetrievalMultiplier = hasMetadataFilters ? 4 : 1;
        int lexicalK = query.LexicalK * overRetrievalMultiplier;
        int vectorK = query.VectorK * overRetrievalMultiplier;

        var rankedLists = new List<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)>();

        // Lexical retrieval with field boosts
        if (query.Text is not null)
        {
            var lexSw = Stopwatch.StartNew();
            var lexicalResults = _lexicalRetriever.Search(query.Text, lexicalK, query.TitleBoost, query.BodyBoost);
            lexSw.Stop();
            lexicalTimeMs = lexSw.Elapsed.TotalMilliseconds;

            if (lexicalResults.Count > 0)
            {
                rankedLists.Add((lexicalResults, query.LexicalWeight, "lexical"));
            }
        }

        // Vector retrieval
        if (queryVector is not null && _vectorRetriever.Count > 0)
        {
            var vecSw = Stopwatch.StartNew();
            var vectorResults = _vectorRetriever.Search(queryVector, vectorK);
            vecSw.Stop();
            vectorTimeMs = vecSw.Elapsed.TotalMilliseconds;

            if (vectorResults.Count > 0)
            {
                rankedLists.Add((vectorResults, query.VectorWeight, "vector"));
            }
        }

        // Fusion
        IReadOnlyList<SearchResult> results;
        if (rankedLists.Count == 0)
        {
            results = Array.Empty<SearchResult>();
        }
        else
        {
            var fuseSw = Stopwatch.StartNew();
            // When filtering, fuse more candidates than TopK so we have enough after filtering
            int fuseTopK = hasMetadataFilters ? query.TopK * overRetrievalMultiplier : query.TopK;
            results = _fuser.Fuse(rankedLists, query.RrfK, fuseTopK, query.Explain);
            fuseSw.Stop();
            fusionTimeMs = fuseSw.Elapsed.TotalMilliseconds;
        }

        // Metadata filtering (post-fusion)
        if (hasMetadataFilters && results.Count > 0)
        {
            var filterSw = Stopwatch.StartNew();
            var filters = query.MetadataFilters!;
            var filtered = new List<SearchResult>();

            foreach (var result in results)
            {
                if (_documents.TryGetValue(result.Id, out var doc) && doc.Metadata is not null)
                {
                    bool matches = true;
                    foreach (var (key, value) in filters)
                    {
                        if (!doc.Metadata.TryGetValue(key, out var docValue) ||
                            !string.Equals(docValue, value, StringComparison.Ordinal))
                        {
                            matches = false;
                            break;
                        }
                    }

                    if (matches)
                        filtered.Add(result);
                }

                if (filtered.Count >= query.TopK)
                    break;
            }

            results = filtered;
            filterSw.Stop();
            filterTimeMs = filterSw.Elapsed.TotalMilliseconds;
        }

        totalSw.Stop();

        var timing = new QueryTimingBreakdown
        {
            LexicalTimeMs = lexicalTimeMs,
            VectorTimeMs = vectorTimeMs,
            FusionTimeMs = fusionTimeMs,
            EmbeddingTimeMs = embeddingTimeMs,
            FilterTimeMs = filterTimeMs,
            TotalTimeMs = totalSw.Elapsed.TotalMilliseconds
        };

        return new SearchResponse
        {
            Results = results,
            QueryTimeMs = totalSw.Elapsed.TotalMilliseconds,
            TimingBreakdown = timing
        };
    }

    private static void ValidateBoosts(HybridQuery query)
    {
        if (float.IsNaN(query.TitleBoost) || float.IsInfinity(query.TitleBoost) || query.TitleBoost < 0)
            throw new ArgumentOutOfRangeException(nameof(query), $"TitleBoost must be a finite non-negative value, got {query.TitleBoost}.");
        if (float.IsNaN(query.BodyBoost) || float.IsInfinity(query.BodyBoost) || query.BodyBoost < 0)
            throw new ArgumentOutOfRangeException(nameof(query), $"BodyBoost must be a finite non-negative value, got {query.BodyBoost}.");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _lexicalRetriever.Dispose();
            _disposed = true;
        }
    }
}
