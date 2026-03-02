using System.Diagnostics;
using HybridSearch.Abstractions;
using HybridSearch.Fusion;
using HybridSearch.Lexical;
using HybridSearch.Models;
using HybridSearch.Vector;

namespace HybridSearch;

/// <summary>
/// Fluent builder for constructing a <see cref="HybridSearchIndex"/>.
/// Supports adding documents directly, from collections, or by loading files from disk.
/// </summary>
public sealed class HybridSearchIndexBuilder
{
    private readonly List<Document> _documents = new();
    private IEmbeddingProvider? _embeddingProvider;
    private IFuser? _fuser;

    /// <summary>
    /// Add a single document to the index.
    /// </summary>
    public HybridSearchIndexBuilder AddDocument(Document doc)
    {
        ArgumentNullException.ThrowIfNull(doc);
        _documents.Add(doc);
        return this;
    }

    /// <summary>
    /// Add multiple documents to the index.
    /// </summary>
    public HybridSearchIndexBuilder AddDocuments(IEnumerable<Document> docs)
    {
        ArgumentNullException.ThrowIfNull(docs);
        _documents.AddRange(docs);
        return this;
    }

    /// <summary>
    /// Load text files from a folder and add them as documents.
    /// Supports *.md and *.txt files. The file name (without extension) becomes the title,
    /// and a stable hash-based ID is generated from the relative path.
    /// </summary>
    /// <param name="folderPath">Path to the folder to scan.</param>
    /// <param name="searchPattern">File search pattern (default: "*.*" to find all text files).</param>
    /// <param name="recursive">Whether to scan subdirectories.</param>
    public HybridSearchIndexBuilder AddFolder(string folderPath, string searchPattern = "*.*", bool recursive = true)
    {
        ArgumentNullException.ThrowIfNull(folderPath);

        if (!Directory.Exists(folderPath))
            throw new DirectoryNotFoundException($"Folder not found: {folderPath}");

        var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var supportedExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".md", ".txt" };

        var files = Directory.EnumerateFiles(folderPath, searchPattern, searchOption)
            .Where(f => supportedExtensions.Contains(Path.GetExtension(f)))
            .OrderBy(f => f, StringComparer.Ordinal);

        foreach (var file in files)
        {
            var relativePath = Path.GetRelativePath(folderPath, file);
            var body = File.ReadAllText(file);

            if (string.IsNullOrWhiteSpace(body))
                continue;

            _documents.Add(new Document
            {
                Id = $"file:{relativePath.Replace('\\', '/')}",
                Title = Path.GetFileNameWithoutExtension(file),
                Body = body,
                Metadata = new Dictionary<string, string>
                {
                    ["sourcePath"] = file,
                    ["relativePath"] = relativePath,
                    ["extension"] = Path.GetExtension(file)
                }
            });
        }

        return this;
    }

    /// <summary>
    /// Configure an embedding provider for automatic query embedding.
    /// </summary>
    public HybridSearchIndexBuilder WithEmbeddingProvider(IEmbeddingProvider provider)
    {
        _embeddingProvider = provider ?? throw new ArgumentNullException(nameof(provider));
        return this;
    }

    /// <summary>
    /// Configure a custom fuser (defaults to <see cref="RrfFuser"/> if not specified).
    /// </summary>
    public HybridSearchIndexBuilder WithFuser(IFuser fuser)
    {
        _fuser = fuser ?? throw new ArgumentNullException(nameof(fuser));
        return this;
    }

    /// <summary>
    /// Build the index synchronously. Documents with pre-computed embeddings are indexed directly.
    /// Documents without embeddings will have them generated if an embedding provider is configured.
    /// </summary>
    public HybridSearchIndex Build()
    {
        return BuildAsync(CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Build the index asynchronously. Preferred when an embedding provider is configured
    /// to avoid blocking on async embedding calls.
    /// </summary>
    public async Task<HybridSearchIndex> BuildAsync(CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();

        if (_documents.Count == 0)
            throw new InvalidOperationException("Cannot build an index with no documents.");

        // Generate embeddings for documents that don't have them
        await EnsureEmbeddingsAsync(ct).ConfigureAwait(false);

        var lexicalRetriever = new LuceneLexicalRetriever();
        var vectorRetriever = new BruteForceVectorRetriever();
        var fuser = _fuser ?? new RrfFuser();
        var docMap = new Dictionary<string, Document>(_documents.Count);

        int? embeddingDimension = null;

        foreach (var doc in _documents)
        {
            docMap[doc.Id] = doc;

            // Index in lexical retriever
            lexicalRetriever.Add(doc.Id, doc.Body);

            // Index in vector retriever (if embedding available)
            if (doc.Embedding is not null)
            {
                vectorRetriever.Add(doc.Id, doc.Embedding);
                embeddingDimension ??= doc.Embedding.Length;
            }
        }

        sw.Stop();

        var stats = new IndexStats
        {
            DocumentCount = _documents.Count,
            EmbeddingDimension = embeddingDimension,
            IndexBuildTimeMs = sw.Elapsed.TotalMilliseconds
        };

        return new HybridSearchIndex(lexicalRetriever, vectorRetriever, fuser, _embeddingProvider, docMap, stats);
    }

    /// <summary>
    /// Ensure all documents have embeddings. If a provider is configured,
    /// batch-embed documents that are missing embeddings.
    /// </summary>
    private async Task EnsureEmbeddingsAsync(CancellationToken ct)
    {
        if (_embeddingProvider is null)
            return;

        var docsNeedingEmbedding = _documents
            .Select((doc, index) => (doc, index))
            .Where(x => x.doc.Embedding is null)
            .ToList();

        if (docsNeedingEmbedding.Count == 0)
            return;

        var texts = docsNeedingEmbedding.Select(x => x.doc.Body).ToList();
        var embeddings = await _embeddingProvider.EmbedBatchAsync(texts, ct).ConfigureAwait(false);

        for (int i = 0; i < docsNeedingEmbedding.Count; i++)
        {
            var (doc, originalIndex) = docsNeedingEmbedding[i];
            // Replace with a new Document that includes the embedding
            _documents[originalIndex] = new Document
            {
                Id = doc.Id,
                Title = doc.Title,
                Body = doc.Body,
                Embedding = embeddings[i],
                Metadata = doc.Metadata
            };
        }
    }
}
