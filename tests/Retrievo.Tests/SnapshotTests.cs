using System.Text;
using Retrievo.Abstractions;
using Retrievo.Fusion;
using Retrievo.Models;
using Retrievo.Snapshots;
using Retrievo.Vector;
using NSubstitute;

namespace Retrievo.Tests;

public class SnapshotTests
{
    [Fact]
    public void ExportSnapshot_ThenImportSnapshot_RestoresSearchBehavior()
    {
        var docs = new[]
        {
            new Document
            {
                Id = "doc-1",
                Title = "Neural Systems",
                Body = "Neural networks learn complex patterns from training data.",
                Embedding = new float[] { 1f, 0f },
                Metadata = new Dictionary<string, string> { ["tags"] = "ml|ai" }
            },
            new Document
            {
                Id = "doc-2",
                Title = "Database Systems",
                Body = "Relational databases store structured records for analytics.",
                Embedding = new float[] { 0f, 1f },
                Metadata = new Dictionary<string, string> { ["tags"] = "data|sql" }
            }
        };

        using var original = new HybridSearchIndexBuilder()
            .DefineField("tags", FieldType.StringArray)
            .AddDocuments(docs)
            .Build();

        using var stream = new MemoryStream();
        original.ExportSnapshot(stream);
        stream.Position = 0;

        using var restored = HybridSearchIndex.ImportSnapshot(stream);

        var query = new HybridQuery
        {
            Text = "neural network",
            Vector = docs[0].Embedding,
            TopK = 5,
            MetadataFilters = new Dictionary<string, string> { ["tags"] = "ml" }
        };

        var originalResponse = original.Search(query);
        var restoredResponse = restored.Search(query);

        Assert.Equal(originalResponse.Results.Select(result => result.Id), restoredResponse.Results.Select(result => result.Id));
        Assert.Equal(originalResponse.Results[0].Score, restoredResponse.Results[0].Score, precision: 6);

        var stats = restored.GetStats();
        Assert.Equal(2, stats.DocumentCount);
        Assert.Equal(2, stats.EmbeddingDimension);
    }

    [Fact]
    public void ExportSnapshot_FromMutableIndex_OnlyIncludesCommittedDocuments()
    {
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "committed",
            Body = "alpha beta gamma"
        });
        index.Commit();

        index.Upsert(new Document
        {
            Id = "pending",
            Body = "delta epsilon zeta"
        });

        using var stream = new MemoryStream();
        index.ExportSnapshot(stream);
        stream.Position = 0;

        using var restored = HybridSearchIndex.ImportSnapshot(stream);

        var committed = restored.Search(new HybridQuery { Text = "alpha", TopK = 5 });
        var pending = restored.Search(new HybridQuery { Text = "delta", TopK = 5 });

        Assert.Single(committed.Results);
        Assert.Equal("committed", committed.Results[0].Id);
        Assert.Empty(pending.Results);
    }

    [Fact]
    public void ImportSnapshot_IntoMutableIndex_AllowsFurtherMutations()
    {
        using var original = new HybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Body = "alpha beta gamma"
            })
            .Build();

        using var stream = new MemoryStream();
        original.ExportSnapshot(stream);
        stream.Position = 0;

        using var restored = MutableHybridSearchIndex.ImportSnapshot(stream);

        restored.Upsert(new Document
        {
            Id = "doc-2",
            Body = "delta epsilon zeta"
        });
        restored.Commit();

        var response = restored.Search(new HybridQuery { Text = "delta", TopK = 5 });

        Assert.Single(response.Results);
        Assert.Equal("doc-2", response.Results[0].Id);
    }

    [Fact]
    public async Task ExportSnapshotAsync_ThenImportSnapshotAsync_PreservesEmbeddingProviderUsage()
    {
        var docs = new[]
        {
            new Document
            {
                Id = "doc-1",
                Body = "neural retrieval systems",
                Embedding = new float[] { 1f, 0f }
            },
            new Document
            {
                Id = "doc-2",
                Body = "database indexing",
                Embedding = new float[] { 0f, 1f }
            }
        };

        using var original = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var provider = Substitute.For<IEmbeddingProvider>();
        provider.Dimensions.Returns(2);
        provider.EmbedAsync("neural retrieval", Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(new float[] { 1f, 0f }));
        provider.EmbedBatchAsync(Arg.Any<IReadOnlyList<string>>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(Array.Empty<float[]>()));

        await using var stream = new MemoryStream();
        await original.ExportSnapshotAsync(stream);
        stream.Position = 0;

        using var restored = await HybridSearchIndex.ImportSnapshotAsync(stream, provider);

        var response = await restored.SearchAsync(new HybridQuery
        {
            Text = "neural retrieval",
            TopK = 5
        });

        Assert.NotEmpty(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
        await provider.Received(1).EmbedAsync("neural retrieval", Arg.Any<CancellationToken>());
    }

    [Fact]
    public void ImportSnapshot_UnsupportedVersion_Throws()
    {
        const string payload = """
            {
              "format": "retrievo-snapshot",
              "formatVersion": 999,
              "createdUtc": "2026-03-06T00:00:00+00:00",
              "documents": [],
              "fieldDefinitions": []
            }
            """;

        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(payload));

        var ex = Assert.Throws<NotSupportedException>(() => HybridSearchIndex.ImportSnapshot(stream));

        Assert.Contains("version 999", ex.Message);
    }

    [Fact]
    public void ImportSnapshot_CustomFuserWithoutOverride_Throws()
    {
        var customFuser = new ReverseOrdinalFuser();

        using var original = new HybridSearchIndexBuilder()
            .AddDocument(new Document { Id = "alpha", Body = "shared term" })
            .AddDocument(new Document { Id = "beta", Body = "shared term" })
            .WithFuser(customFuser)
            .Build();

        using var stream = new MemoryStream();
        original.ExportSnapshot(stream);
        stream.Position = 0;

        var ex = Assert.Throws<InvalidOperationException>(() => HybridSearchIndex.ImportSnapshot(stream));

        Assert.Contains("custom fuser", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ImportSnapshot_CustomFuserOverride_PreservesRankingBehavior()
    {
        var customFuser = new ReverseOrdinalFuser();

        using var original = new HybridSearchIndexBuilder()
            .AddDocument(new Document { Id = "alpha", Body = "shared term" })
            .AddDocument(new Document { Id = "beta", Body = "shared term" })
            .WithFuser(customFuser)
            .Build();

        using var stream = new MemoryStream();
        original.ExportSnapshot(stream);
        stream.Position = 0;

        using var restored = HybridSearchIndex.ImportSnapshot(stream, fuser: customFuser);

        var originalResponse = original.Search(new HybridQuery { Text = "shared", TopK = 5 });
        var restoredResponse = restored.Search(new HybridQuery { Text = "shared", TopK = 5 });

        Assert.Equal("beta", originalResponse.Results[0].Id);
        Assert.Equal(originalResponse.Results.Select(result => result.Id), restoredResponse.Results.Select(result => result.Id));
    }

    [Fact]
    public void SnapshotSerializer_RoundTrip_PreservesEmbeddingBitsExactly()
    {
        var originalEmbedding = new[]
        {
            BitConverter.Int32BitsToSingle(unchecked((int)0x3F800001)),
            BitConverter.Int32BitsToSingle(unchecked((int)0xBF123456)),
            BitConverter.Int32BitsToSingle(unchecked((int)0x3EAAAAAB))
        };
        var normalizedEmbedding = VectorMath.Normalize(originalEmbedding);

        var documents = new[]
        {
            new Document
            {
                Id = "doc-1",
                Body = "precision test",
                Embedding = originalEmbedding
            }
        };

        using var stream = new MemoryStream();
        IndexSnapshotSerializer.Write(
            stream,
            documents,
            [("doc-1", normalizedEmbedding)],
            new Dictionary<string, FieldDefinition>(StringComparer.Ordinal),
            new RrfFuser());
        stream.Position = 0;

        var snapshot = IndexSnapshotSerializer.Read(stream);
        var restoredDocument = Assert.Single(snapshot.Documents);
        var restoredVectorEntry = Assert.Single(snapshot.VectorEntries!);

        Assert.Null(restoredDocument.Embedding);
        Assert.Equal("doc-1", restoredVectorEntry.Id);
        Assert.Equal(normalizedEmbedding.Length, restoredVectorEntry.NormalizedEmbedding.Length);

        for (var i = 0; i < normalizedEmbedding.Length; i++)
        {
            Assert.Equal(
                BitConverter.SingleToInt32Bits(normalizedEmbedding[i]),
                BitConverter.SingleToInt32Bits(restoredVectorEntry.NormalizedEmbedding[i]));
        }
    }

    [Fact]
    public void ImportSnapshot_Version2EmbeddingArrayPayload_RemainsSupported()
    {
        const string payload = """
            {
              "format": "retrievo-snapshot",
              "formatVersion": 2,
              "createdUtc": "2026-03-06T00:00:00+00:00",
              "documents": [
                {
                  "id": "doc-1",
                  "body": "legacy payload",
                  "embedding": [1.0, 0.5]
                }
              ],
              "fieldDefinitions": [],
              "fuser": {
                "kind": "Rrf"
              }
            }
            """;

        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(payload));
        using var restored = HybridSearchIndex.ImportSnapshot(stream);

        var stats = restored.GetStats();
        Assert.Equal(1, stats.DocumentCount);
        Assert.Equal(2, stats.EmbeddingDimension);

        var response = restored.Search(new HybridQuery
        {
            Vector = new float[] { 1f, 0.5f },
            TopK = 10
        });

        Assert.Single(response.Results);
        Assert.Equal("doc-1", response.Results[0].Id);
    }

    [Fact]
    public void SnapshotDocumentOrderer_PreservesVectorEntryOrder()
    {
        var documents = new Dictionary<string, Document>(StringComparer.Ordinal)
        {
            ["doc-1"] = new Document { Id = "doc-1", Body = "one", Embedding = new float[] { 1f, 0f } },
            ["doc-2"] = new Document { Id = "doc-2", Body = "two" },
            ["doc-3"] = new Document { Id = "doc-3", Body = "three", Embedding = new float[] { 0f, 1f } }
        };
        var vectorEntries = new (string Id, float[] NormalizedEmbedding)[]
        {
            ("doc-3", new float[] { 0f, 1f }),
            ("doc-1", new float[] { 1f, 0f })
        };

        var ordered = SnapshotDocumentOrderer.Order(documents, vectorEntries);

        Assert.Equal(["doc-3", "doc-1", "doc-2"], ordered.Select(document => document.Id));
    }

    private sealed class ReverseOrdinalFuser : IFuser
    {
        public IReadOnlyList<SearchResult> Fuse(
            IReadOnlyList<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)> rankedLists,
            int rrfK,
            int topK,
            bool explain)
        {
            return rankedLists
                .SelectMany(list => list.Items)
                .Select(item => item.Id)
                .Distinct(StringComparer.Ordinal)
                .OrderByDescending(id => id, StringComparer.Ordinal)
                .Take(topK)
                .Select((id, index) => new SearchResult
                {
                    Id = id,
                    Score = topK - index
                })
                .ToArray();
        }
    }
}
