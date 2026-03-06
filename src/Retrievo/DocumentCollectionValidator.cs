using Retrievo.Models;

namespace Retrievo;

/// <summary>
/// Shared validation helpers for document collections supplied to builders.
/// </summary>
internal static class DocumentCollectionValidator
{
    /// <summary>
    /// Ensures all document identifiers are unique within a single build.
    /// </summary>
    /// <param name="documents">The documents queued for indexing.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the collection contains duplicate document identifiers.
    /// </exception>
    internal static void ValidateUniqueIds(IEnumerable<Document> documents)
    {
        ArgumentNullException.ThrowIfNull(documents);

        var seenIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (var document in documents)
        {
            ArgumentNullException.ThrowIfNull(document);
            ArgumentNullException.ThrowIfNull(document.Id);

            if (!seenIds.Add(document.Id))
            {
                throw new InvalidOperationException(
                    $"Duplicate document ID '{document.Id}'. Document IDs must be unique within a single index build.");
            }
        }
    }
}
