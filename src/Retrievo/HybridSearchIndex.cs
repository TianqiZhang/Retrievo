using System.Diagnostics;
using Retrievo.Abstractions;
using Retrievo.Fusion;
using Retrievo.Lexical;
using Retrievo.Models;
using Retrievo.Vector;

namespace Retrievo;

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
        query.ValidateBoosts();

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
        query.ValidateBoosts();

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

        bool hasExactFilters = query.MetadataFilters is not null && query.MetadataFilters.Count > 0;
        bool hasRangeFilters = query.MetadataRangeFilters is not null && query.MetadataRangeFilters.Count > 0;
        bool hasContainsFilters = query.MetadataContainsFilters is not null && query.MetadataContainsFilters.Count > 0;
        bool hasMetadataFilters = hasExactFilters || hasRangeFilters || hasContainsFilters;

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
            var filtered = new List<SearchResult>();

            foreach (var result in results)
            {
                if (_documents.TryGetValue(result.Id, out var doc) && doc.Metadata is not null)
                {
                    if (MatchesAllFilters(doc, query))
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

    /// <summary>
    /// Checks whether a document matches all configured metadata filters (exact, range, and contains).
    /// </summary>
    private static bool MatchesAllFilters(Document doc, HybridQuery query)
    {
        var metadata = doc.Metadata!;

        // Exact-match filters
        if (query.MetadataFilters is not null)
        {
            foreach (var (key, value) in query.MetadataFilters)
            {
                if (!metadata.TryGetValue(key, out var docValue) ||
                    !string.Equals(docValue, value, StringComparison.Ordinal))
                    return false;
            }
        }

        // Range filters (ordinal string comparison — works for ISO 8601 and zero-padded numbers)
        if (query.MetadataRangeFilters is not null)
        {
            foreach (var filter in query.MetadataRangeFilters)
            {
                if (!metadata.TryGetValue(filter.Key, out var docValue))
                    return false;

                if (filter.Min is not null && string.Compare(docValue, filter.Min, StringComparison.Ordinal) < 0)
                    return false;

                if (filter.Max is not null && string.Compare(docValue, filter.Max, StringComparison.Ordinal) > 0)
                    return false;
            }
        }

        // Contains filters (split metadata value by delimiter, check if any element matches)
        if (query.MetadataContainsFilters is not null)
        {
            foreach (var (key, value) in query.MetadataContainsFilters)
            {
                if (!metadata.TryGetValue(key, out var docValue))
                    return false;

                bool found = false;
                foreach (var segment in docValue.Split(query.MetadataContainsDelimiter))
                {
                    if (string.Equals(segment, value, StringComparison.Ordinal))
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                    return false;
            }
        }

        return true;
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
