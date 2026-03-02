using HybridSearch.Abstractions;
using HybridSearch.Models;
using Lucene.Net.Index;
using Lucene.Net.QueryParsers.Classic;
using Lucene.Net.Search;
using Lucene.Net.Search.Similarities;
using Lucene.Net.Store;
using Lucene.Net.Util;
using LuceneDocument = Lucene.Net.Documents.Document;
using LuceneField = Lucene.Net.Documents.Field;
using LuceneStringField = Lucene.Net.Documents.StringField;
using LuceneTextField = Lucene.Net.Documents.TextField;

namespace HybridSearch.Lexical;

/// <summary>
/// Lexical retriever backed by Lucene.NET's in-memory RAMDirectory with BM25 scoring.
/// Documents are indexed by their title and body text fields and searched via parsed text queries.
/// Supports per-field boost weights at query time.
/// </summary>
public sealed class LuceneLexicalRetriever : ILexicalRetriever
{
    private const string FieldId = "id";
    private const string FieldTitle = "title";
    private const string FieldBody = "body";
    private const LuceneVersion Version = LuceneVersion.LUCENE_48;

    private readonly RAMDirectory _directory;
    private readonly LuceneTextAnalyzer _textAnalyzer;
    private readonly IndexWriter _writer;
    private IndexSearcher? _searcher;
    private DirectoryReader? _reader;
    private bool _dirty = true;
    private bool _disposed;

    /// <summary>
    /// When true, the searcher is only refreshed on explicit <see cref="RefreshSearcher"/> calls,
    /// not automatically on each search. Used by <see cref="MutableHybridSearchIndex"/> for snapshot isolation.
    /// </summary>
    internal bool ManualRefreshOnly { get; set; }

    /// <summary>
    /// Number of documents in the index.
    /// </summary>
    public int Count => _writer.NumDocs;

    /// <summary>
    /// Creates a new LuceneLexicalRetriever with a default StandardAnalyzer.
    /// </summary>
    public LuceneLexicalRetriever()
        : this(new LuceneTextAnalyzer())
    {
    }

    /// <summary>
    /// Creates a new LuceneLexicalRetriever with the specified text analyzer.
    /// </summary>
    public LuceneLexicalRetriever(LuceneTextAnalyzer textAnalyzer)
    {
        _textAnalyzer = textAnalyzer ?? throw new ArgumentNullException(nameof(textAnalyzer));
        _directory = new RAMDirectory();
        var config = new IndexWriterConfig(Version, _textAnalyzer.Analyzer)
        {
            OpenMode = OpenMode.CREATE
        };
        _writer = new IndexWriter(_directory, config);
    }

    /// <summary>
    /// Add a document to the lexical index.
    /// </summary>
    /// <param name="id">The unique document ID.</param>
    /// <param name="body">The document body text to index.</param>
    /// <param name="title">Optional document title to index.</param>
    public void Add(string id, string body, string? title = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(id);
        ArgumentNullException.ThrowIfNull(body);

        var doc = new LuceneDocument
        {
            new LuceneStringField(FieldId, id, LuceneField.Store.YES),
            new LuceneTextField(FieldBody, body, LuceneField.Store.NO)
        };

        if (!string.IsNullOrWhiteSpace(title))
        {
            doc.Add(new LuceneTextField(FieldTitle, title, LuceneField.Store.NO));
        }

        _writer.AddDocument(doc);
        _dirty = true;
    }

    /// <summary>
    /// Update an existing document or add it if it doesn't exist.
    /// Uses the document ID as the unique key for replacement.
    /// </summary>
    /// <param name="id">The unique document ID.</param>
    /// <param name="body">The document body text to index.</param>
    /// <param name="title">Optional document title to index.</param>
    public void Update(string id, string body, string? title = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(id);
        ArgumentNullException.ThrowIfNull(body);

        var doc = new LuceneDocument
        {
            new LuceneStringField(FieldId, id, LuceneField.Store.YES),
            new LuceneTextField(FieldBody, body, LuceneField.Store.NO)
        };

        if (!string.IsNullOrWhiteSpace(title))
        {
            doc.Add(new LuceneTextField(FieldTitle, title, LuceneField.Store.NO));
        }

        var term = new Term(FieldId, id);
        _writer.UpdateDocument(term, doc);
        _dirty = true;
    }

    /// <summary>
    /// Delete a document from the lexical index by its ID.
    /// </summary>
    /// <param name="id">The document ID to delete.</param>
    /// <returns>True (deletion is always queued; actual removal happens on next commit/refresh).</returns>
    public bool Delete(string id)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(id);

        var term = new Term(FieldId, id);
        _writer.DeleteDocuments(term);
        _dirty = true;
        return true;
    }

    /// <summary>
    /// Explicitly refresh the Lucene searcher to reflect pending writes.
    /// Used by <see cref="MutableHybridSearchIndex.Commit"/> to make changes visible.
    /// </summary>
    internal void RefreshSearcher()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _writer.Commit();

        var newReader = _reader is null
            ? DirectoryReader.Open(_directory)
            : DirectoryReader.OpenIfChanged(_reader);

        if (newReader is not null)
        {
            if (!ManualRefreshOnly)
            {
                // In automatic mode (immutable index), dispose old reader immediately — no concurrent writers.
                _reader?.Dispose();
            }
            // In ManualRefreshOnly mode (mutable index), old reader is kept alive by snapshot ref-counting.
            // MutableHybridSearchIndex manages old reader lifecycle via AcquireSearcherSnapshot/ReleaseSearcherSnapshot.
            _reader = newReader;
            _searcher = new IndexSearcher(_reader);
            _searcher.Similarity = new BM25Similarity();
        }

        _dirty = false;
    }

    /// <summary>
    /// Acquire the current IndexSearcher and its underlying DirectoryReader for snapshot use.
    /// The caller is responsible for calling <see cref="ReleaseSearcherSnapshot"/> when done.
    /// The DirectoryReader's ref count is incremented to prevent premature disposal.
    /// </summary>
    internal (IndexSearcher Searcher, DirectoryReader Reader)? AcquireSearcherSnapshot()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_searcher is null || _reader is null)
            return null;

        _reader.IncRef();
        return (_searcher, _reader);
    }

    /// <summary>
    /// Release a previously acquired searcher snapshot by decrementing the reader's ref count.
    /// When the ref count reaches zero, the reader is closed automatically by Lucene.
    /// </summary>
    internal static void ReleaseSearcherSnapshot(DirectoryReader reader)
    {
        reader.DecRef();
    }

    /// <summary>
    /// Ensure the searcher is refreshed to reflect any pending writes.
    /// </summary>
    private void EnsureSearcher()
    {
        if (ManualRefreshOnly)
        {
            // In manual mode, only refresh if no searcher exists at all (initial state)
            if (_searcher is null)
            {
                RefreshSearcher();
            }
            return;
        }

        if (_dirty || _searcher is null)
        {
            RefreshSearcher();
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<RankedItem> Search(string text, int topK)
    {
        return Search(text, topK, titleBoost: 1f, bodyBoost: 1f);
    }

    /// <summary>
    /// Search the lexical index with per-field boost weights.
    /// </summary>
    /// <param name="text">The text query.</param>
    /// <param name="topK">Maximum number of results to return.</param>
    /// <param name="titleBoost">Boost weight for the title field.</param>
    /// <param name="bodyBoost">Boost weight for the body field.</param>
    /// <returns>A ranked list of document IDs with scores, ordered by descending relevance.</returns>
    public IReadOnlyList<RankedItem> Search(string text, int topK, float titleBoost, float bodyBoost)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);

        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<RankedItem>();

        EnsureSearcher();

        if (_searcher is null || _reader!.NumDocs == 0)
            return Array.Empty<RankedItem>();

        var escapedText = QueryParser.Escape(text);
        Query query;

        try
        {
            // Build per-field queries with individual boosts
            var bodyParser = new QueryParser(Version, FieldBody, _textAnalyzer.Analyzer);
            var bodyQuery = bodyParser.Parse(escapedText);

            var titleParser = new QueryParser(Version, FieldTitle, _textAnalyzer.Analyzer);
            var titleQuery = titleParser.Parse(escapedText);

            var boolQuery = new BooleanQuery();

            bodyQuery.Boost = bodyBoost;
            boolQuery.Add(bodyQuery, Occur.SHOULD);

            titleQuery.Boost = titleBoost;
            boolQuery.Add(titleQuery, Occur.SHOULD);

            query = boolQuery;
        }
        catch (ParseException)
        {
            return Array.Empty<RankedItem>();
        }

        var topDocs = _searcher.Search(query, topK);
        var results = new RankedItem[topDocs.ScoreDocs.Length];

        for (int i = 0; i < topDocs.ScoreDocs.Length; i++)
        {
            var scoreDoc = topDocs.ScoreDocs[i];
            var storedDoc = _searcher.Doc(scoreDoc.Doc);
            var docId = storedDoc.Get(FieldId);

            results[i] = new RankedItem
            {
                Id = docId,
                Score = scoreDoc.Score,
                Rank = i + 1 // 1-based
            };
        }

        return results;
    }

    /// <summary>
    /// Search using an externally-provided IndexSearcher (from a snapshot).
    /// This allows snapshot-isolated lexical searches in <see cref="MutableHybridSearchIndex"/>.
    /// </summary>
    internal IReadOnlyList<RankedItem> SearchWithSearcher(string text, int topK, float titleBoost, float bodyBoost, IndexSearcher searcher)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(searcher);

        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<RankedItem>();

        var escapedText = QueryParser.Escape(text);
        Query query;

        try
        {
            var bodyParser = new QueryParser(Version, FieldBody, _textAnalyzer.Analyzer);
            var bodyQuery = bodyParser.Parse(escapedText);

            var titleParser = new QueryParser(Version, FieldTitle, _textAnalyzer.Analyzer);
            var titleQuery = titleParser.Parse(escapedText);

            var boolQuery = new BooleanQuery();

            bodyQuery.Boost = bodyBoost;
            boolQuery.Add(bodyQuery, Occur.SHOULD);

            titleQuery.Boost = titleBoost;
            boolQuery.Add(titleQuery, Occur.SHOULD);

            query = boolQuery;
        }
        catch (ParseException)
        {
            return Array.Empty<RankedItem>();
        }

        var topDocs = searcher.Search(query, topK);
        var results = new RankedItem[topDocs.ScoreDocs.Length];

        for (int i = 0; i < topDocs.ScoreDocs.Length; i++)
        {
            var scoreDoc = topDocs.ScoreDocs[i];
            var storedDoc = searcher.Doc(scoreDoc.Doc);
            var docId = storedDoc.Get(FieldId);

            results[i] = new RankedItem
            {
                Id = docId,
                Score = scoreDoc.Score,
                Rank = i + 1
            };
        }

        return results;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _reader?.Dispose();
            _writer.Dispose();
            _directory.Dispose();
            _textAnalyzer.Dispose();
            _disposed = true;
        }
    }
}
