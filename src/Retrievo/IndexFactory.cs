using System.Diagnostics;
using Retrievo.Abstractions;
using Retrievo.Fusion;
using Retrievo.Lexical;
using Retrievo.Models;
using Retrievo.Vector;

namespace Retrievo;

/// <summary>
/// Shared index construction helpers used by builders and snapshot import.
/// </summary>
internal static class IndexFactory
{
    internal static HybridSearchIndex CreateHybridSearchIndex(
        IReadOnlyList<Document> documents,
        IEmbeddingProvider? embeddingProvider,
        IFuser? fuser,
        IReadOnlyDictionary<string, FieldDefinition> fieldDefinitions,
        double baseElapsedMs,
        bool allowEmptyDocuments = false,
        IReadOnlyList<(string Id, float[] NormalizedEmbedding)>? vectorEntries = null)
    {
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(fieldDefinitions);

        var sw = Stopwatch.StartNew();

        if (!allowEmptyDocuments && documents.Count == 0)
            throw new InvalidOperationException("Cannot build an index with no documents.");

        DocumentCollectionValidator.ValidateUniqueIds(documents);

        var artifacts = BuildArtifacts(documents, vectorEntries);
        sw.Stop();

        var stats = CreateStats(documents.Count, artifacts.EmbeddingDimension, baseElapsedMs + sw.Elapsed.TotalMilliseconds);
        var fieldDefinitionsCopy = new Dictionary<string, FieldDefinition>(fieldDefinitions, StringComparer.Ordinal);

        return new HybridSearchIndex(
            artifacts.LexicalRetriever,
            artifacts.VectorRetriever,
            fuser ?? new RrfFuser(),
            embeddingProvider,
            artifacts.DocumentMap,
            stats,
            fieldDefinitionsCopy);
    }

    internal static MutableHybridSearchIndex CreateMutableHybridSearchIndex(
        IReadOnlyList<Document> documents,
        IEmbeddingProvider? embeddingProvider,
        IFuser? fuser,
        IReadOnlyDictionary<string, FieldDefinition> fieldDefinitions,
        double baseElapsedMs,
        bool allowEmptyDocuments = true,
        IReadOnlyList<(string Id, float[] NormalizedEmbedding)>? vectorEntries = null)
    {
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(fieldDefinitions);

        var sw = Stopwatch.StartNew();

        if (!allowEmptyDocuments && documents.Count == 0)
            throw new InvalidOperationException("Cannot build an index with no documents.");

        DocumentCollectionValidator.ValidateUniqueIds(documents);

        var artifacts = BuildArtifacts(documents, vectorEntries);
        sw.Stop();

        var stats = CreateStats(documents.Count, artifacts.EmbeddingDimension, baseElapsedMs + sw.Elapsed.TotalMilliseconds);
        var fieldDefinitionsCopy = new Dictionary<string, FieldDefinition>(fieldDefinitions, StringComparer.Ordinal);

        return new MutableHybridSearchIndex(
            artifacts.LexicalRetriever,
            artifacts.VectorRetriever,
            fuser ?? new RrfFuser(),
            embeddingProvider,
            artifacts.DocumentMap,
            stats,
            fieldDefinitionsCopy);
    }

    private static IndexBuildArtifacts BuildArtifacts(
        IReadOnlyList<Document> documents,
        IReadOnlyList<(string Id, float[] NormalizedEmbedding)>? vectorEntries)
    {
        var lexicalRetriever = new LuceneLexicalRetriever();
        var vectorRetriever = new BruteForceVectorRetriever();
        var docMap = new Dictionary<string, Document>(documents.Count, StringComparer.Ordinal);

        foreach (var doc in documents)
        {
            docMap[doc.Id] = doc;
            lexicalRetriever.Add(doc.Id, doc.Body, doc.Title);
        }

        if (vectorEntries is not null)
        {
            foreach (var (id, normalizedEmbedding) in vectorEntries)
            {
                if (!docMap.ContainsKey(id))
                    throw new InvalidOperationException($"Vector entry '{id}' does not have a matching document.");

                vectorRetriever.AddNormalized(id, normalizedEmbedding);
            }
        }
        else
        {
            foreach (var doc in documents)
            {
                if (doc.Embedding is not null)
                    vectorRetriever.Add(doc.Id, doc.Embedding);
            }
        }

        int? embeddingDimension = vectorRetriever.Count == 0 ? null : vectorRetriever.Dimensions;

        return new IndexBuildArtifacts(lexicalRetriever, vectorRetriever, docMap, embeddingDimension);
    }

    private static IndexStats CreateStats(int documentCount, int? embeddingDimension, double buildTimeMs)
    {
        return new IndexStats
        {
            DocumentCount = documentCount,
            EmbeddingDimension = embeddingDimension,
            IndexBuildTimeMs = buildTimeMs
        };
    }

    private sealed record IndexBuildArtifacts(
        LuceneLexicalRetriever LexicalRetriever,
        BruteForceVectorRetriever VectorRetriever,
        Dictionary<string, Document> DocumentMap,
        int? EmbeddingDimension);
}
