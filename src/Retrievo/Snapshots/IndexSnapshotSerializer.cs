using System.Buffers.Binary;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Retrievo.Models;

namespace Retrievo.Snapshots;

/// <summary>
/// Serializes and deserializes logical index snapshots.
/// </summary>
internal static class IndexSnapshotSerializer
{
    private const string SnapshotFormat = "retrievo-snapshot";
    private const int CurrentFormatVersion = 4;

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() }
    };

    internal static void Write(
        Stream stream,
        IEnumerable<Document> documents,
        IReadOnlyList<(string Id, float[] NormalizedEmbedding)> vectorEntries,
        IReadOnlyDictionary<string, FieldDefinition> fieldDefinitions,
        Abstractions.IFuser fuser)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(vectorEntries);
        ArgumentNullException.ThrowIfNull(fieldDefinitions);
        ArgumentNullException.ThrowIfNull(fuser);

        var payload = CreatePayload(documents, vectorEntries, fieldDefinitions, fuser);
        JsonSerializer.Serialize(stream, payload, JsonOptions);
    }

    internal static Task WriteAsync(
        Stream stream,
        IEnumerable<Document> documents,
        IReadOnlyList<(string Id, float[] NormalizedEmbedding)> vectorEntries,
        IReadOnlyDictionary<string, FieldDefinition> fieldDefinitions,
        Abstractions.IFuser fuser,
        CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(vectorEntries);
        ArgumentNullException.ThrowIfNull(fieldDefinitions);
        ArgumentNullException.ThrowIfNull(fuser);

        var payload = CreatePayload(documents, vectorEntries, fieldDefinitions, fuser);
        return JsonSerializer.SerializeAsync(stream, payload, JsonOptions, ct);
    }

    internal static IndexSnapshotData Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var payload = JsonSerializer.Deserialize<PersistedIndexSnapshot>(stream, JsonOptions)
            ?? throw new InvalidDataException("Snapshot payload is empty.");

        return ValidateAndProject(payload);
    }

    internal static async Task<IndexSnapshotData> ReadAsync(Stream stream, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var payload = await JsonSerializer.DeserializeAsync<PersistedIndexSnapshot>(stream, JsonOptions, ct).ConfigureAwait(false)
            ?? throw new InvalidDataException("Snapshot payload is empty.");

        return ValidateAndProject(payload);
    }

    private static PersistedIndexSnapshot CreatePayload(
        IEnumerable<Document> documents,
        IReadOnlyList<(string Id, float[] NormalizedEmbedding)> vectorEntries,
        IReadOnlyDictionary<string, FieldDefinition> fieldDefinitions,
        Abstractions.IFuser fuser)
    {
        return new PersistedIndexSnapshot
        {
            Format = SnapshotFormat,
            FormatVersion = CurrentFormatVersion,
            CreatedUtc = DateTimeOffset.UtcNow,
            Documents = documents.Select(CreatePersistedDocument).ToList(),
            VectorEntries = vectorEntries.Select(CreatePersistedVectorEntry).ToList(),
            FieldDefinitions = fieldDefinitions.Values.ToList(),
            Fuser = SnapshotFuserRegistry.Describe(fuser)
        };
    }

    private static IndexSnapshotData ValidateAndProject(PersistedIndexSnapshot payload)
    {
        if (!string.Equals(payload.Format, SnapshotFormat, StringComparison.Ordinal))
            throw new InvalidDataException($"Unsupported snapshot format '{payload.Format}'.");

        if (payload.FormatVersion is not 1 and not 2 and not 3 and not CurrentFormatVersion)
            throw new NotSupportedException(
                $"Snapshot format version {payload.FormatVersion} is not supported. Supported versions: 1-{CurrentFormatVersion}.");

        var documents = new List<Document>((payload.Documents ?? []).Count);
        var documentIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (var document in payload.Documents ?? [])
        {
            if (document is null)
                throw new InvalidDataException("Snapshot contains a null document entry.");

            if (string.IsNullOrWhiteSpace(document.Id))
                throw new InvalidDataException("Snapshot contains a document with a missing ID.");

            if (document.Body is null)
                throw new InvalidDataException($"Snapshot document '{document.Id}' is missing a body.");

            if (!documentIds.Add(document.Id))
                throw new InvalidDataException($"Snapshot contains duplicate document '{document.Id}'.");

            documents.Add(new Document
            {
                Id = document.Id,
                Title = document.Title,
                Body = document.Body,
                Embedding = payload.FormatVersion switch
                {
                    1 or 2 => document.Embedding,
                    3 => DeserializeEmbedding(document.EmbeddingBase64),
                    _ => null
                },
                Metadata = document.Metadata
            });
        }

        var fieldDefinitions = new Dictionary<string, FieldDefinition>(StringComparer.Ordinal);
        foreach (var definition in payload.FieldDefinitions ?? [])
        {
            if (definition is null)
                throw new InvalidDataException("Snapshot contains a null field definition entry.");

            definition.Validate();

            if (!fieldDefinitions.TryAdd(definition.Name, definition))
            {
                throw new InvalidDataException(
                    $"Snapshot contains duplicate field definition '{definition.Name}'.");
            }
        }

        var fuser = payload.FormatVersion switch
        {
            1 => new SnapshotFuserDescriptor(SnapshotFuserKind.Rrf),
            _ => payload.Fuser ?? throw new InvalidDataException("Snapshot is missing fuser metadata.")
        };

        var vectorEntries = payload.FormatVersion >= 4
            ? ValidateVectorEntries(payload.VectorEntries, documentIds)
            : null;

        return new IndexSnapshotData(documents, fieldDefinitions, fuser, vectorEntries);
    }

    private static PersistedSnapshotDocument CreatePersistedDocument(Document document)
    {
        ArgumentNullException.ThrowIfNull(document);

        return new PersistedSnapshotDocument
        {
            Id = document.Id,
            Title = document.Title,
            Body = document.Body,
            Metadata = document.Metadata is null
                ? null
                : new Dictionary<string, string>(document.Metadata, StringComparer.Ordinal)
        };
    }

    private static PersistedSnapshotVectorEntry CreatePersistedVectorEntry((string Id, float[] NormalizedEmbedding) vectorEntry)
    {
        return new PersistedSnapshotVectorEntry
        {
            Id = vectorEntry.Id,
            NormalizedEmbeddingBase64 = SerializeEmbedding(vectorEntry.NormalizedEmbedding)
                ?? throw new InvalidOperationException("Vector snapshot entry is missing an embedding.")
        };
    }

    private static string? SerializeEmbedding(float[]? embedding)
    {
        if (embedding is null)
            return null;

        var bytes = new byte[embedding.Length * sizeof(float)];
        var byteSpan = bytes.AsSpan();

        for (var i = 0; i < embedding.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(
                byteSpan.Slice(i * sizeof(float), sizeof(float)),
                BitConverter.SingleToInt32Bits(embedding[i]));
        }

        return Convert.ToBase64String(bytes);
    }

    private static float[]? DeserializeEmbedding(string? embeddingBase64)
    {
        if (embeddingBase64 is null)
            return null;

        byte[] bytes;
        try
        {
            bytes = Convert.FromBase64String(embeddingBase64);
        }
        catch (FormatException ex)
        {
            throw new InvalidDataException("Snapshot contains an invalid embedding payload.", ex);
        }

        if (bytes.Length == 0 || bytes.Length % sizeof(float) != 0)
            throw new InvalidDataException("Snapshot contains an invalid embedding payload length.");

        var embedding = new float[bytes.Length / sizeof(float)];
        var byteSpan = bytes.AsSpan();

        for (var i = 0; i < embedding.Length; i++)
        {
            var bits = BinaryPrimitives.ReadInt32LittleEndian(
                byteSpan.Slice(i * sizeof(float), sizeof(float)));
            embedding[i] = BitConverter.Int32BitsToSingle(bits);
        }

        return embedding;
    }

    private static IReadOnlyList<(string Id, float[] NormalizedEmbedding)>? ValidateVectorEntries(
        List<PersistedSnapshotVectorEntry>? vectorEntries,
        HashSet<string> documentIds)
    {
        ArgumentNullException.ThrowIfNull(documentIds);

        if (vectorEntries is null || vectorEntries.Count == 0)
            return null;

        var projectedEntries = new List<(string Id, float[] NormalizedEmbedding)>(vectorEntries.Count);
        var seenIds = new HashSet<string>(StringComparer.Ordinal);

        foreach (var entry in vectorEntries)
        {
            if (entry is null)
                throw new InvalidDataException("Snapshot contains a null vector entry.");

            if (string.IsNullOrWhiteSpace(entry.Id))
                throw new InvalidDataException("Snapshot contains a vector entry with a missing ID.");

            if (!documentIds.Contains(entry.Id))
                throw new InvalidDataException($"Snapshot vector entry '{entry.Id}' does not have a matching document.");

            if (!seenIds.Add(entry.Id))
                throw new InvalidDataException($"Snapshot contains duplicate vector entry '{entry.Id}'.");

            projectedEntries.Add((
                entry.Id,
                DeserializeEmbedding(entry.NormalizedEmbeddingBase64)
                    ?? throw new InvalidDataException($"Snapshot vector entry '{entry.Id}' is missing an embedding.")));
        }

        return projectedEntries;
    }

    private sealed class PersistedIndexSnapshot
    {
        public required string Format { get; init; }

        public int FormatVersion { get; init; }

        public DateTimeOffset CreatedUtc { get; init; }

        public List<PersistedSnapshotDocument>? Documents { get; init; }

        public List<PersistedSnapshotVectorEntry>? VectorEntries { get; init; }

        public List<FieldDefinition>? FieldDefinitions { get; init; }

        public SnapshotFuserDescriptor? Fuser { get; init; }
    }

    private sealed class PersistedSnapshotDocument
    {
        public required string Id { get; init; }

        public string? Title { get; init; }

        public required string Body { get; init; }

        public Dictionary<string, string>? Metadata { get; init; }

        public float[]? Embedding { get; init; }

        public string? EmbeddingBase64 { get; init; }
    }

    private sealed class PersistedSnapshotVectorEntry
    {
        public required string Id { get; init; }

        public required string NormalizedEmbeddingBase64 { get; init; }
    }
}

/// <summary>
/// In-memory representation of a deserialized snapshot.
/// </summary>
internal sealed record IndexSnapshotData(
    IReadOnlyList<Document> Documents,
    IReadOnlyDictionary<string, FieldDefinition> FieldDefinitions,
    SnapshotFuserDescriptor Fuser,
    IReadOnlyList<(string Id, float[] NormalizedEmbedding)>? VectorEntries);
