using System.Diagnostics;
using Retrievo.Abstractions;
using Retrievo.Fusion;
using Retrievo.Models;

namespace Retrievo;

/// <summary>
/// Fluent builder for constructing a <see cref="MutableHybridSearchIndex"/>.
/// Supports seeding the index with initial documents, then allows incremental updates via the index itself.
/// </summary>
public sealed class MutableHybridSearchIndexBuilder
{
    private readonly List<Document> _documents = new();
    private IEmbeddingProvider? _embeddingProvider;
    private IFuser? _fuser;
    private readonly Dictionary<string, FieldDefinition> _fieldDefinitions = new(StringComparer.Ordinal);

    /// <summary>
    /// Add a single document to seed the index.
    /// </summary>
    public MutableHybridSearchIndexBuilder AddDocument(Document doc)
    {
        ArgumentNullException.ThrowIfNull(doc);
        _documents.Add(doc);
        return this;
    }

    /// <summary>
    /// Add multiple documents to seed the index.
    /// </summary>
    public MutableHybridSearchIndexBuilder AddDocuments(IEnumerable<Document> docs)
    {
        ArgumentNullException.ThrowIfNull(docs);
        _documents.AddRange(docs);
        return this;
    }

    /// <summary>
    /// Configure an embedding provider for automatic document and query embedding.
    /// </summary>
    public MutableHybridSearchIndexBuilder WithEmbeddingProvider(IEmbeddingProvider provider)
    {
        _embeddingProvider = provider ?? throw new ArgumentNullException(nameof(provider));
        return this;
    }

    /// <summary>
    /// Configure a custom fuser (defaults to <see cref="RrfFuser"/> if not specified).
    /// </summary>
    public MutableHybridSearchIndexBuilder WithFuser(IFuser fuser)
    {
        _fuser = fuser ?? throw new ArgumentNullException(nameof(fuser));
        return this;
    }

    /// <summary>
    /// Declare a metadata field type. Fields not declared default to <see cref="FieldType.String"/> (exact match).
    /// Use <see cref="FieldType.StringArray"/> for delimited multi-value fields.
    /// </summary>
    /// <param name="name">Metadata key this definition applies to.</param>
    /// <param name="type">The field type that determines filter behavior.</param>
    /// <param name="delimiter">Delimiter for <see cref="FieldType.StringArray"/> fields (default '|').</param>
    public MutableHybridSearchIndexBuilder DefineField(string name, FieldType type, char delimiter = '|')
    {
        ArgumentNullException.ThrowIfNull(name);
        var def = new FieldDefinition { Name = name, Type = type, Delimiter = delimiter };
        def.Validate();
        _fieldDefinitions[name] = def;
        return this;
    }

    /// <summary>
    /// Build the mutable index synchronously. Seed documents are indexed and an initial commit is created.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown when an embedding provider is configured and one or more seed documents are missing pre-computed embeddings.
    /// Use <see cref="BuildAsync(CancellationToken)"/>, or provide pre-computed embeddings on all documents.
    /// </exception>
    public MutableHybridSearchIndex Build()
    {
        var sw = Stopwatch.StartNew();
        DocumentCollectionValidator.ValidateUniqueIds(_documents);

        if (_embeddingProvider is not null && _documents.Any(doc => doc.Embedding is null))
            throw new InvalidOperationException("Synchronous Build() cannot generate embeddings for documents. Use BuildAsync(), or provide pre-computed Embedding values on all documents.");

        sw.Stop();
        return IndexFactory.CreateMutableHybridSearchIndex(
            _documents,
            _embeddingProvider,
            _fuser,
            _fieldDefinitions,
            sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Build the mutable index asynchronously. Preferred when an embedding provider is configured.
    /// </summary>
    public async Task<MutableHybridSearchIndex> BuildAsync(CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        DocumentCollectionValidator.ValidateUniqueIds(_documents);

        // Generate embeddings for documents that don't have them
        await EnsureEmbeddingsAsync(ct).ConfigureAwait(false);

        sw.Stop();
        return IndexFactory.CreateMutableHybridSearchIndex(
            _documents,
            _embeddingProvider,
            _fuser,
            _fieldDefinitions,
            sw.Elapsed.TotalMilliseconds);
    }

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
