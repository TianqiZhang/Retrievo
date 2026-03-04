using System.Diagnostics;
using Retrievo.Abstractions;
using Retrievo.Fusion;
using Retrievo.Lexical;
using Retrievo.Models;
using Retrievo.Vector;
using Lucene.Net.Index;
using Lucene.Net.Search;

namespace Retrievo;

/// <summary>
/// A mutable hybrid search index that supports incremental updates (upsert, delete, commit).
/// Readers always see the last committed snapshot; writes are buffered until <see cref="Commit"/> is called.
/// Thread-safe for concurrent reads. Writes require external synchronization (single-writer discipline).
/// </summary>
public sealed class MutableHybridSearchIndex : IMutableHybridSearchIndex
{
    private readonly LuceneLexicalRetriever _lexicalRetriever;
    private readonly BruteForceVectorRetriever _vectorRetriever;
    private readonly IFuser _fuser;
    private readonly IEmbeddingProvider? _embeddingProvider;

    // Live (mutable) document map — updated on Upsert/Delete
    private readonly Dictionary<string, Document> _pendingDocuments;

    // Snapshot visible to readers — swapped atomically on Commit()
    private volatile SearchSnapshot _snapshot;

    private bool _disposed;

    internal MutableHybridSearchIndex(
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
        _pendingDocuments = documents;

        // Enable manual refresh mode — only refresh Lucene searcher on Commit()
        _lexicalRetriever.ManualRefreshOnly = true;

        // Initial commit to create the first readable snapshot
        _lexicalRetriever.RefreshSearcher();
        _snapshot = CreateSnapshot(stats);
    }

    /// <inheritdoc/>
    public void Upsert(Document doc)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(doc);

        _lexicalRetriever.Update(doc.Id, doc.Body, doc.Title);

        if (doc.Embedding is not null)
        {
            _vectorRetriever.Update(doc.Id, doc.Embedding);
        }
        else
        {
            // Remove stale vector entry if the new doc has no embedding
            _vectorRetriever.Remove(doc.Id);
        }

        _pendingDocuments[doc.Id] = doc;
    }

    /// <inheritdoc/>
    public async Task UpsertAsync(Document doc, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(doc);

        var embedding = doc.Embedding;
        if (embedding is null && _embeddingProvider is not null)
        {
            embedding = await _embeddingProvider.EmbedAsync(doc.Body, ct).ConfigureAwait(false);
            doc = new Document
            {
                Id = doc.Id,
                Title = doc.Title,
                Body = doc.Body,
                Embedding = embedding,
                Metadata = doc.Metadata
            };
        }

        _lexicalRetriever.Update(doc.Id, doc.Body, doc.Title);

        if (doc.Embedding is not null)
        {
            _vectorRetriever.Update(doc.Id, doc.Embedding);
        }
        else
        {
            _vectorRetriever.Remove(doc.Id);
        }

        _pendingDocuments[doc.Id] = doc;
    }

    /// <inheritdoc/>
    public bool Delete(string id)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(id);

        if (!_pendingDocuments.Remove(id))
            return false;

        _lexicalRetriever.Delete(id);
        _vectorRetriever.Remove(id);
        return true;
    }

    /// <inheritdoc/>
    public void Commit()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Capture old snapshot before refresh so we can release its reader
        var oldSnapshot = _snapshot;

        // Refresh the Lucene searcher to see all pending writes
        _lexicalRetriever.RefreshSearcher();

        var stats = new IndexStats
        {
            DocumentCount = _pendingDocuments.Count,
            EmbeddingDimension = _vectorRetriever.Dimensions > 0 ? _vectorRetriever.Dimensions : null,
            IndexBuildTimeMs = 0
        };

        // Atomically swap the snapshot
        _snapshot = CreateSnapshot(stats);

        // Release old snapshot's reader (DecRef — when refcount hits 0, Lucene closes it)
        if (oldSnapshot.Reader is not null)
        {
            LuceneLexicalRetriever.ReleaseSearcherSnapshot(oldSnapshot.Reader);
        }
    }

    /// <inheritdoc/>
    public SearchResponse Search(HybridQuery query)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(query);
        query.ValidateBoosts();

        var totalSw = Stopwatch.StartNew();
        var snapshot = _snapshot; // Read volatile once

        float[]? queryVector = query.Vector;
        double? embeddingTimeMs = null;
        if (queryVector is null && query.Text is not null && _embeddingProvider is not null)
        {
            var embedSw = Stopwatch.StartNew();
            queryVector = _embeddingProvider.EmbedAsync(query.Text).GetAwaiter().GetResult();
            embedSw.Stop();
            embeddingTimeMs = embedSw.Elapsed.TotalMilliseconds;
        }

        return ExecuteSearch(query, queryVector, embeddingTimeMs, snapshot, totalSw);
    }

    /// <inheritdoc/>
    public async Task<SearchResponse> SearchAsync(HybridQuery query, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(query);
        query.ValidateBoosts();

        var totalSw = Stopwatch.StartNew();
        var snapshot = _snapshot; // Read volatile once

        float[]? queryVector = query.Vector;
        double? embeddingTimeMs = null;
        if (queryVector is null && query.Text is not null && _embeddingProvider is not null)
        {
            var embedSw = Stopwatch.StartNew();
            queryVector = await _embeddingProvider.EmbedAsync(query.Text, ct).ConfigureAwait(false);
            embedSw.Stop();
            embeddingTimeMs = embedSw.Elapsed.TotalMilliseconds;
        }

        return ExecuteSearch(query, queryVector, embeddingTimeMs, snapshot, totalSw);
    }

    /// <inheritdoc/>
    public IndexStats GetStats()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _snapshot.Stats;
    }

    private SearchResponse ExecuteSearch(HybridQuery query, float[]? queryVector, double? embeddingTimeMs, SearchSnapshot snapshot, Stopwatch totalSw)
    {
        double? lexicalTimeMs = null;
        double? vectorTimeMs = null;
        double fusionTimeMs = 0;
        double? filterTimeMs = null;

        bool hasExactFilters = query.MetadataFilters is not null && query.MetadataFilters.Count > 0;
        bool hasRangeFilters = query.MetadataRangeFilters is not null && query.MetadataRangeFilters.Count > 0;
        bool hasContainsFilters = query.MetadataContainsFilters is not null && query.MetadataContainsFilters.Count > 0;
        bool hasMetadataFilters = hasExactFilters || hasRangeFilters || hasContainsFilters;

        int overRetrievalMultiplier = hasMetadataFilters ? 4 : 1;
        int lexicalK = query.LexicalK * overRetrievalMultiplier;
        int vectorK = query.VectorK * overRetrievalMultiplier;

        var rankedLists = new List<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)>();

        // Lexical retrieval from snapshot-isolated searcher
        if (query.Text is not null && snapshot.Searcher is not null)
        {
            var lexSw = Stopwatch.StartNew();
            var lexicalResults = _lexicalRetriever.SearchWithSearcher(query.Text, lexicalK, query.TitleBoost, query.BodyBoost, snapshot.Searcher);
            lexSw.Stop();
            lexicalTimeMs = lexSw.Elapsed.TotalMilliseconds;

            if (lexicalResults.Count > 0)
            {
                rankedLists.Add((lexicalResults, query.LexicalWeight, "lexical"));
            }
        }

        // Vector retrieval from snapshot
        if (queryVector is not null && snapshot.VectorEntries.Count > 0)
        {
            var vecSw = Stopwatch.StartNew();
            var vectorResults = SearchVectorSnapshot(queryVector, vectorK, snapshot.VectorEntries);
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
                if (snapshot.Documents.TryGetValue(result.Id, out var doc) && doc.Metadata is not null)
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

    private static IReadOnlyList<RankedItem> SearchVectorSnapshot(
        float[] queryVector, int topK, IReadOnlyList<(string Id, float[] NormalizedEmbedding)> entries)
    {
        if (entries.Count == 0)
            return Array.Empty<RankedItem>();

        var normalizedQuery = VectorMath.Normalize(queryVector);

        var scored = new (string Id, float Similarity)[entries.Count];
        for (int i = 0; i < entries.Count; i++)
        {
            var (id, embedding) = entries[i];
            float sim = VectorMath.DotProduct(normalizedQuery, embedding);
            scored[i] = (id, sim);
        }

        Array.Sort(scored, (a, b) =>
        {
            int cmp = b.Similarity.CompareTo(a.Similarity);
            return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
        });

        int resultCount = Math.Min(topK, scored.Length);
        var results = new RankedItem[resultCount];
        for (int i = 0; i < resultCount; i++)
        {
            results[i] = new RankedItem
            {
                Id = scored[i].Id,
                Score = scored[i].Similarity,
                Rank = i + 1
            };
        }

        return results;
    }

    private SearchSnapshot CreateSnapshot(IndexStats stats)
    {
        // Snapshot the document map (shallow copy — Document is immutable)
        var docsCopy = new Dictionary<string, Document>(_pendingDocuments, StringComparer.Ordinal);

        // Snapshot vector entries from the retriever
        var vectorEntries = _vectorRetriever.GetSnapshotEntries();

        // Acquire snapshot-isolated Lucene searcher (IncRef prevents premature reader disposal)
        var searcherSnapshot = _lexicalRetriever.AcquireSearcherSnapshot();
        var searcher = searcherSnapshot?.Searcher;
        var reader = searcherSnapshot?.Reader;

        return new SearchSnapshot(docsCopy, vectorEntries, stats, searcher, reader);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            // Release the current snapshot's reader before disposing the retriever
            if (_snapshot.Reader is not null)
            {
                LuceneLexicalRetriever.ReleaseSearcherSnapshot(_snapshot.Reader);
            }

            _lexicalRetriever.Dispose();
            _disposed = true;
        }
    }

    /// <summary>
    /// Immutable snapshot of index state at the time of Commit().
    /// Includes the Lucene IndexSearcher/DirectoryReader for snapshot-isolated lexical search.
    /// </summary>
    private sealed class SearchSnapshot
    {
        public IReadOnlyDictionary<string, Document> Documents { get; }
        public IReadOnlyList<(string Id, float[] NormalizedEmbedding)> VectorEntries { get; }
        public IndexStats Stats { get; }
        public IndexSearcher? Searcher { get; }
        public DirectoryReader? Reader { get; }

        public SearchSnapshot(
            Dictionary<string, Document> documents,
            IReadOnlyList<(string Id, float[] NormalizedEmbedding)> vectorEntries,
            IndexStats stats,
            IndexSearcher? searcher,
            DirectoryReader? reader)
        {
            Documents = documents;
            VectorEntries = vectorEntries;
            Stats = stats;
            Searcher = searcher;
            Reader = reader;
        }
    }
}
