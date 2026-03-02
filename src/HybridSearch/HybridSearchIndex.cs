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

        // If text is provided, no vector, and we have an embedding provider, embed synchronously
        float[]? queryVector = query.Vector;
        if (queryVector is null && query.Text is not null && _embeddingProvider is not null)
        {
            queryVector = _embeddingProvider.EmbedAsync(query.Text).GetAwaiter().GetResult();
        }

        return ExecuteSearch(query, queryVector);
    }

    /// <inheritdoc/>
    public async Task<SearchResponse> SearchAsync(HybridQuery query, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(query);

        // If text is provided, no vector, and we have an embedding provider, embed asynchronously
        float[]? queryVector = query.Vector;
        if (queryVector is null && query.Text is not null && _embeddingProvider is not null)
        {
            queryVector = await _embeddingProvider.EmbedAsync(query.Text, ct).ConfigureAwait(false);
        }

        return ExecuteSearch(query, queryVector);
    }

    /// <inheritdoc/>
    public IndexStats GetStats()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _stats;
    }

    private SearchResponse ExecuteSearch(HybridQuery query, float[]? queryVector)
    {
        var sw = Stopwatch.StartNew();

        var rankedLists = new List<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)>();

        // Lexical retrieval
        if (query.Text is not null)
        {
            var lexicalResults = _lexicalRetriever.Search(query.Text, query.LexicalK);
            if (lexicalResults.Count > 0)
            {
                rankedLists.Add((lexicalResults, query.LexicalWeight, "lexical"));
            }
        }

        // Vector retrieval
        if (queryVector is not null && _vectorRetriever.Count > 0)
        {
            var vectorResults = _vectorRetriever.Search(queryVector, query.VectorK);
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
            results = _fuser.Fuse(rankedLists, query.RrfK, query.TopK, query.Explain);
        }

        sw.Stop();

        return new SearchResponse
        {
            Results = results,
            QueryTimeMs = sw.Elapsed.TotalMilliseconds
        };
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
