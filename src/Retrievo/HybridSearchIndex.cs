using System.Diagnostics;
using Retrievo.Abstractions;
using Retrievo.Fusion;
using Retrievo.Lexical;
using Retrievo.Models;
using Retrievo.Snapshots;
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
    private readonly IReadOnlyDictionary<string, FieldDefinition> _fieldDefinitions;
    private bool _disposed;

    internal HybridSearchIndex(
        LuceneLexicalRetriever lexicalRetriever,
        BruteForceVectorRetriever vectorRetriever,
        IFuser fuser,
        IEmbeddingProvider? embeddingProvider,
        Dictionary<string, Document> documents,
        IndexStats stats,
        Dictionary<string, FieldDefinition> fieldDefinitions)
    {
        _lexicalRetriever = lexicalRetriever;
        _vectorRetriever = vectorRetriever;
        _fuser = fuser;
        _embeddingProvider = embeddingProvider;
        _documents = documents;
        _stats = stats;
        _fieldDefinitions = fieldDefinitions;
    }

    /// <summary>
    /// Execute a hybrid search query synchronously.
    /// </summary>
    /// <param name="query">The hybrid query to execute.</param>
    /// <returns>The search response containing fused results and timing information.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="query"/> is null.</exception>
    /// <exception cref="ObjectDisposedException">Thrown when this index has been disposed.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when query text requires embedding generation and an embedding provider is configured.
    /// Use <see cref="SearchAsync(HybridQuery, CancellationToken)"/>, or provide a pre-computed vector in <see cref="HybridQuery.Vector"/>.
    /// </exception>
    public SearchResponse Search(HybridQuery query)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(query);
        query.ValidateBoosts();

        var totalSw = Stopwatch.StartNew();

        // If text is provided, no vector, and we have an embedding provider, sync path cannot embed
        float[]? queryVector = query.Vector;
        double? embeddingTimeMs = null;
        if (queryVector is null && query.Text is not null && _embeddingProvider is not null)
        {
            throw new InvalidOperationException("Synchronous Search() cannot generate embeddings for query text. Use SearchAsync(), or provide a pre-computed Vector in the HybridQuery.");
        }

        return ExecuteSearch(query, queryVector, embeddingTimeMs, totalSw, CancellationToken.None);
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

        return ExecuteSearch(query, queryVector, embeddingTimeMs, totalSw, ct);
    }

    /// <inheritdoc/>
    public IndexStats GetStats()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _stats;
    }

    /// <summary>
    /// Export the index contents to a versioned JSON snapshot file.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    public void ExportSnapshot(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ObjectDisposedException.ThrowIf(_disposed, this);

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        ExportSnapshot(stream);
    }

    /// <summary>
    /// Export the index contents to a versioned JSON snapshot file asynchronously.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task ExportSnapshotAsync(string path, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ObjectDisposedException.ThrowIf(_disposed, this);

        await using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 4096,
            useAsync: true);

        await ExportSnapshotAsync(stream, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Export the index contents to a versioned JSON snapshot stream.
    /// </summary>
    /// <param name="stream">Destination stream.</param>
    public void ExportSnapshot(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ObjectDisposedException.ThrowIf(_disposed, this);

        var vectorEntries = _vectorRetriever.GetSnapshotEntries();
        var orderedDocuments = SnapshotDocumentOrderer.Order(_documents, vectorEntries);
        IndexSnapshotSerializer.Write(stream, orderedDocuments, vectorEntries, _fieldDefinitions, _fuser);
    }

    /// <summary>
    /// Export the index contents to a versioned JSON snapshot stream asynchronously.
    /// </summary>
    /// <param name="stream">Destination stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public Task ExportSnapshotAsync(Stream stream, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ObjectDisposedException.ThrowIf(_disposed, this);

        var vectorEntries = _vectorRetriever.GetSnapshotEntries();
        var orderedDocuments = SnapshotDocumentOrderer.Order(_documents, vectorEntries);
        return IndexSnapshotSerializer.WriteAsync(stream, orderedDocuments, vectorEntries, _fieldDefinitions, _fuser, ct);
    }

    /// <summary>
    /// Import a snapshot file into a new immutable index.
    /// </summary>
    /// <param name="path">Snapshot file path.</param>
    /// <param name="embeddingProvider">
    /// Optional embedding provider used only for future query embedding. Stored document embeddings come from the snapshot.
    /// </param>
    /// <param name="fuser">
    /// Optional fuser override. Required when importing a snapshot created with a custom <see cref="IFuser"/>.
    /// </param>
    public static HybridSearchIndex ImportSnapshot(
        string path,
        IEmbeddingProvider? embeddingProvider = null,
        IFuser? fuser = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        return ImportSnapshot(stream, embeddingProvider, fuser);
    }

    /// <summary>
    /// Import a snapshot file into a new immutable index asynchronously.
    /// </summary>
    /// <param name="path">Snapshot file path.</param>
    /// <param name="embeddingProvider">
    /// Optional embedding provider used only for future query embedding. Stored document embeddings come from the snapshot.
    /// </param>
    /// <param name="fuser">
    /// Optional fuser override. Required when importing a snapshot created with a custom <see cref="IFuser"/>.
    /// </param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task<HybridSearchIndex> ImportSnapshotAsync(
        string path,
        IEmbeddingProvider? embeddingProvider = null,
        IFuser? fuser = null,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        await using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 4096,
            useAsync: true);

        return await ImportSnapshotAsync(stream, embeddingProvider, fuser, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Import a snapshot stream into a new immutable index.
    /// </summary>
    /// <param name="stream">Snapshot stream.</param>
    /// <param name="embeddingProvider">
    /// Optional embedding provider used only for future query embedding. Stored document embeddings come from the snapshot.
    /// </param>
    /// <param name="fuser">
    /// Optional fuser override. Required when importing a snapshot created with a custom <see cref="IFuser"/>.
    /// </param>
    public static HybridSearchIndex ImportSnapshot(
        Stream stream,
        IEmbeddingProvider? embeddingProvider = null,
        IFuser? fuser = null)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var sw = Stopwatch.StartNew();
        var snapshot = IndexSnapshotSerializer.Read(stream);
        sw.Stop();

        return IndexFactory.CreateHybridSearchIndex(
            snapshot.Documents,
            embeddingProvider,
            SnapshotFuserRegistry.Resolve(snapshot.Fuser, fuser),
            snapshot.FieldDefinitions,
            sw.Elapsed.TotalMilliseconds,
            allowEmptyDocuments: true,
            vectorEntries: snapshot.VectorEntries);
    }

    /// <summary>
    /// Import a snapshot stream into a new immutable index asynchronously.
    /// </summary>
    /// <param name="stream">Snapshot stream.</param>
    /// <param name="embeddingProvider">
    /// Optional embedding provider used only for future query embedding. Stored document embeddings come from the snapshot.
    /// </param>
    /// <param name="fuser">
    /// Optional fuser override. Required when importing a snapshot created with a custom <see cref="IFuser"/>.
    /// </param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task<HybridSearchIndex> ImportSnapshotAsync(
        Stream stream,
        IEmbeddingProvider? embeddingProvider = null,
        IFuser? fuser = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var sw = Stopwatch.StartNew();
        var snapshot = await IndexSnapshotSerializer.ReadAsync(stream, ct).ConfigureAwait(false);
        sw.Stop();

        return IndexFactory.CreateHybridSearchIndex(
            snapshot.Documents,
            embeddingProvider,
            SnapshotFuserRegistry.Resolve(snapshot.Fuser, fuser),
            snapshot.FieldDefinitions,
            sw.Elapsed.TotalMilliseconds,
            allowEmptyDocuments: true,
            vectorEntries: snapshot.VectorEntries);
    }

    private SearchResponse ExecuteSearch(HybridQuery query, float[]? queryVector, double? embeddingTimeMs, Stopwatch totalSw, CancellationToken ct)
    {
        double? lexicalTimeMs = null;
        double? vectorTimeMs = null;
        double fusionTimeMs = 0;
        double? filterTimeMs = null;

        bool hasExactFilters = query.MetadataFilters is not null && query.MetadataFilters.Count > 0;
        bool hasRangeFilters = query.MetadataRangeFilters is not null && query.MetadataRangeFilters.Count > 0;
        bool hasMetadataFilters = hasExactFilters || hasRangeFilters;

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
            var vectorResults = _vectorRetriever.Search(queryVector, vectorK, ct);
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
                    if (MetadataFilterEvaluator.MatchesAll(doc.Metadata, query, _fieldDefinitions))
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
