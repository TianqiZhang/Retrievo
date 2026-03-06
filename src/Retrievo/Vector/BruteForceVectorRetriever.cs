using System.Numerics.Tensors;
using Retrievo.Abstractions;
using Retrievo.Models;

namespace Retrievo.Vector;

/// <summary>
/// Brute-force vector retrieval using cosine similarity.
/// Stores all embeddings in memory as pre-normalized float arrays.
/// Uses SIMD-accelerated dot product for the similarity scan.
/// </summary>
public sealed class BruteForceVectorRetriever : IVectorRetriever
{
    private readonly List<(string Id, float[] NormalizedEmbedding)> _entries = new();

    /// <summary>
    /// Validates that a vector contains only finite values (no NaN or Infinity).
    /// Uses SIMD-accelerated dot-self to propagate NaN/Infinity in O(1) checks.
    /// </summary>
    private static void ValidateFiniteValues(float[] embedding, string paramName)
    {
        // Dot(v, v) propagates NaN and produces Infinity for Infinity inputs.
        // A single float.IsFinite check covers all elements.
        float dotSelf = TensorPrimitives.Dot(embedding, embedding);
        if (!float.IsFinite(dotSelf))
            throw new ArgumentException("Vector contains non-finite values (NaN or Infinity).", paramName);
    }

    /// <summary>
    /// Get a snapshot copy of all entries for use in snapshot-based search.
    /// </summary>
    internal IReadOnlyList<(string Id, float[] NormalizedEmbedding)> GetSnapshotEntries()
    {
        return _entries.ToList();
    }

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

        ValidateFiniteValues(embedding, nameof(embedding));

        if (Dimensions == 0)
            Dimensions = embedding.Length;
        else if (embedding.Length != Dimensions)
            throw new ArgumentException(
                $"Embedding dimension mismatch: expected {Dimensions}, got {embedding.Length}.", nameof(embedding));

        var normalized = VectorMath.Normalize(embedding);
        _entries.Add((id, normalized));
    }

    /// <summary>
    /// Update an existing embedding or add it if not found.
    /// The embedding is normalized on insert.
    /// </summary>
    public void Update(string id, float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(id);
        ArgumentNullException.ThrowIfNull(embedding);

        if (embedding.Length == 0)
            throw new ArgumentException("Embedding must not be empty.", nameof(embedding));

        ValidateFiniteValues(embedding, nameof(embedding));

        if (Dimensions == 0)
            Dimensions = embedding.Length;
        else if (embedding.Length != Dimensions)
            throw new ArgumentException(
                $"Embedding dimension mismatch: expected {Dimensions}, got {embedding.Length}.", nameof(embedding));

        var normalized = VectorMath.Normalize(embedding);

        for (int i = 0; i < _entries.Count; i++)
        {
            if (string.Equals(_entries[i].Id, id, StringComparison.Ordinal))
            {
                _entries[i] = (id, normalized);
                return;
            }
        }

        _entries.Add((id, normalized));
    }

    /// <summary>
    /// Remove a document embedding by its ID.
    /// </summary>
    /// <returns>True if the document was found and removed; false otherwise.</returns>
    public bool Remove(string id)
    {
        ArgumentNullException.ThrowIfNull(id);

        for (int i = 0; i < _entries.Count; i++)
        {
            if (string.Equals(_entries[i].Id, id, StringComparison.Ordinal))
            {
                _entries.RemoveAt(i);
                if (_entries.Count == 0)
                    Dimensions = 0;
                return true;
            }
        }

        return false;
    }

    /// <inheritdoc/>
    public IReadOnlyList<RankedItem> Search(float[] vector, int topK)
    {
        return Search(vector, topK, CancellationToken.None);
    }

    /// <inheritdoc/>
    public IReadOnlyList<RankedItem> Search(float[] vector, int topK, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(vector);

        ValidateFiniteValues(vector, nameof(vector));

        if (_entries.Count == 0)
            return Array.Empty<RankedItem>();

        if (vector.Length != Dimensions)
            throw new ArgumentException(
                $"Query vector dimension mismatch: expected {Dimensions}, got {vector.Length}.", nameof(vector));

        var normalizedQuery = VectorMath.Normalize(vector);

        // Min-heap top-K selection: O(n log k) instead of O(n log n) full sort
        ct.ThrowIfCancellationRequested();

        int k = Math.Min(topK, _entries.Count);
        var heap = new (string Id, float Similarity)[k];
        int heapSize = 0;

        for (int i = 0; i < _entries.Count; i++)
        {
            if ((i & 0xFF) == 0) // every 256 iterations
                ct.ThrowIfCancellationRequested();

            var (id, embedding) = _entries[i];
            float sim = VectorMath.DotProduct(normalizedQuery, embedding);
            var item = (id, sim);

            if (heapSize < k)
            {
                heap[heapSize] = item;
                heapSize++;
                if (heapSize == k)
                    BuildMinHeap(heap, k);
            }
            else if (CompareDescending(item, heap[0]) < 0)
            {
                // item is better (higher similarity) than heap minimum
                heap[0] = item;
                SiftDown(heap, 0, k);
            }
        }

        ct.ThrowIfCancellationRequested();

        // Sort the final heap for deterministic output order
        Array.Sort(heap, 0, heapSize, Comparer<(string Id, float Similarity)>.Create(
            (a, b) =>
            {
                int cmp = b.Similarity.CompareTo(a.Similarity);
                return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
            }));

        var results = new RankedItem[heapSize];
        for (int i = 0; i < heapSize; i++)
        {
            results[i] = new RankedItem
            {
                Id = heap[i].Id,
                Score = heap[i].Similarity,
                Rank = i + 1 // 1-based
            };
        }

        return results;
    }

    // ── Min-heap helpers (ascending by similarity = min-heap for "keep largest K") ──

    /// <summary>
    /// Compare for descending similarity order.
    /// Returns negative if a should come BEFORE b in descending order (a has higher similarity).
    /// </summary>
    private static int CompareDescending((string Id, float Similarity) a, (string Id, float Similarity) b)
    {
        int cmp = b.Similarity.CompareTo(a.Similarity);
        return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
    }

    /// <summary>
    /// Compare for ascending similarity order (used as min-heap comparator).
    /// </summary>
    private static int CompareAscending((string Id, float Similarity) a, (string Id, float Similarity) b)
    {
        int cmp = a.Similarity.CompareTo(b.Similarity);
        return cmp != 0 ? cmp : string.Compare(b.Id, a.Id, StringComparison.Ordinal);
    }

    private static void BuildMinHeap((string Id, float Similarity)[] heap, int size)
    {
        for (int i = size / 2 - 1; i >= 0; i--)
            SiftDown(heap, i, size);
    }

    private static void SiftDown((string Id, float Similarity)[] heap, int i, int size)
    {
        while (true)
        {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int smallest = i;

            if (left < size && CompareAscending(heap[left], heap[smallest]) < 0)
                smallest = left;
            if (right < size && CompareAscending(heap[right], heap[smallest]) < 0)
                smallest = right;

            if (smallest == i)
                break;

            (heap[i], heap[smallest]) = (heap[smallest], heap[i]);
            i = smallest;
        }
    }
}
