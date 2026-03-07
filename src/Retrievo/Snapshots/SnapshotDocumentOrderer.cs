using Retrievo.Models;

namespace Retrievo.Snapshots;

/// <summary>
/// Produces a stable snapshot document order that preserves the live vector entry order.
/// </summary>
internal static class SnapshotDocumentOrderer
{
    internal static IReadOnlyList<Document> Order(
        IReadOnlyDictionary<string, Document> documents,
        IReadOnlyList<(string Id, float[] NormalizedEmbedding)> vectorEntries)
    {
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(vectorEntries);

        if (documents.Count == 0)
            return [];

        if (vectorEntries.Count == 0)
            return documents.Values.ToList();

        var orderedDocuments = new List<Document>(documents.Count);
        var seenIds = new HashSet<string>(StringComparer.Ordinal);

        foreach (var (id, _) in vectorEntries)
        {
            if (!documents.TryGetValue(id, out var document))
            {
                throw new InvalidOperationException(
                    $"Snapshot export encountered vector entry '{id}' without a matching document.");
            }

            if (seenIds.Add(id))
                orderedDocuments.Add(document);
        }

        foreach (var document in documents.Values)
        {
            if (seenIds.Add(document.Id))
                orderedDocuments.Add(document);
        }

        return orderedDocuments;
    }
}
