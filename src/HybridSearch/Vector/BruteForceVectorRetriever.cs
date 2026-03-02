using HybridSearch.Abstractions;
using HybridSearch.Models;

namespace HybridSearch.Vector;

/// <summary>
/// Brute-force vector retrieval using cosine similarity.
/// Stores all embeddings in memory as pre-normalized float arrays.
/// Uses SIMD-accelerated dot product for the similarity scan.
/// </summary>
public sealed class BruteForceVectorRetriever : IVectorRetriever
{
    private readonly List<(string Id, float[] NormalizedEmbedding)> _entries = new();

    /// <summary>
    /// The expected embedding dimension, or 0 if no embeddings have been added yet.
    /// </summary>
    public int Dimensions { get; private set; }

    /// <summary>
    /// Number of indexed embeddings.
    /// </summary>
    public int Count => _entries.Count;

    /// <summary>
    /// Add a document embedding to the index. The embedding is normalized on insert.
    /// </summary>
    public void Add(string id, float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(id);
        ArgumentNullException.ThrowIfNull(embedding);

        if (embedding.Length == 0)
            throw new ArgumentException("Embedding must not be empty.", nameof(embedding));

        if (Dimensions == 0)
            Dimensions = embedding.Length;
        else if (embedding.Length != Dimensions)
            throw new ArgumentException(
                $"Embedding dimension mismatch: expected {Dimensions}, got {embedding.Length}.", nameof(embedding));

        var normalized = VectorMath.Normalize(embedding);
        _entries.Add((id, normalized));
    }

    /// <summary>
    /// Search for the nearest documents to the given query vector.
    /// The query vector is normalized before comparison.
    /// Returns results ranked by cosine similarity (descending).
    /// </summary>
    public IReadOnlyList<RankedItem> Search(float[] vector, int topK)
    {
        ArgumentNullException.ThrowIfNull(vector);

        if (_entries.Count == 0)
            return Array.Empty<RankedItem>();

        if (vector.Length != Dimensions)
            throw new ArgumentException(
                $"Query vector dimension mismatch: expected {Dimensions}, got {vector.Length}.", nameof(vector));

        var normalizedQuery = VectorMath.Normalize(vector);

        // Compute similarities — brute-force scan
        var scored = new (string Id, float Similarity)[_entries.Count];
        for (int i = 0; i < _entries.Count; i++)
        {
            var (id, embedding) = _entries[i];
            float sim = VectorMath.DotProduct(normalizedQuery, embedding);
            scored[i] = (id, sim);
        }

        // Sort by descending similarity, then ordinal Id for tie-break
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
                Rank = i + 1 // 1-based
            };
        }

        return results;
    }
}
