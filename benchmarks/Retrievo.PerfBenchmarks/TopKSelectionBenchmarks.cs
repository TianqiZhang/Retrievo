using System.Buffers;
using BenchmarkDotNet.Attributes;

namespace Retrievo.PerfBenchmarks;

/// <summary>
/// Benchmarks comparing sorting strategies for top-K selection in vector search.
/// The current implementation uses Array.Sort on all entries, then takes top-K.
/// Alternatives: min-heap partial sort, partial quickselect.
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class TopKSelectionBenchmarks
{
    private (string Id, float Similarity)[] _sourceData = null!;

    [Params(1000, 5000, 10000)]
    public int EntryCount { get; set; }

    [Params(10, 50)]
    public int TopK { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _sourceData = new (string, float)[EntryCount];
        for (int i = 0; i < EntryCount; i++)
        {
            _sourceData[i] = ($"doc-{i:D6}", (float)(rng.NextDouble() * 2 - 1));
        }
    }

    /// <summary>
    /// Current implementation: full Array.Sort + take top-K.
    /// </summary>
    [Benchmark(Baseline = true)]
    public (string Id, float Similarity)[] FullSort_Current()
    {
        var scored = new (string Id, float Similarity)[_sourceData.Length];
        Array.Copy(_sourceData, scored, _sourceData.Length);

        Array.Sort(scored, (a, b) =>
        {
            int cmp = b.Similarity.CompareTo(a.Similarity);
            return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
        });

        int resultCount = Math.Min(TopK, scored.Length);
        var results = new (string Id, float Similarity)[resultCount];
        Array.Copy(scored, results, resultCount);
        return results;
    }

    /// <summary>
    /// Min-heap approach: maintain a heap of size K, O(n log k) vs O(n log n).
    /// </summary>
    [Benchmark]
    public (string Id, float Similarity)[] MinHeap_TopK()
    {
        int k = Math.Min(TopK, _sourceData.Length);
        // Use a min-heap of size k: we keep the k largest elements
        var heap = new (string Id, float Similarity)[k];
        int heapSize = 0;

        for (int i = 0; i < _sourceData.Length; i++)
        {
            var item = _sourceData[i];

            if (heapSize < k)
            {
                heap[heapSize] = item;
                heapSize++;
                if (heapSize == k)
                    BuildMinHeap(heap, k);
            }
            else if (CompareDescending(item, heap[0]) < 0)
            {
                // item is "better" (higher similarity) than heap min
                heap[0] = item;
                SiftDown(heap, 0, k);
            }
        }

        // Sort the heap for deterministic output order
        Array.Sort(heap, 0, heapSize, Comparer<(string Id, float Similarity)>.Create(
            (a, b) =>
            {
                int cmp = b.Similarity.CompareTo(a.Similarity);
                return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
            }));

        return heap[..heapSize];
    }

    /// <summary>
    /// ArrayPool variant: same full sort but rents the working array from ArrayPool.
    /// </summary>
    [Benchmark]
    public (string Id, float Similarity)[] FullSort_ArrayPool()
    {
        var pool = ArrayPool<(string Id, float Similarity)>.Shared;
        var scored = pool.Rent(_sourceData.Length);
        try
        {
            Array.Copy(_sourceData, scored, _sourceData.Length);

            Array.Sort(scored, 0, _sourceData.Length, Comparer<(string Id, float Similarity)>.Create(
                (a, b) =>
                {
                    int cmp = b.Similarity.CompareTo(a.Similarity);
                    return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
                }));

            int resultCount = Math.Min(TopK, _sourceData.Length);
            var results = new (string Id, float Similarity)[resultCount];
            Array.Copy(scored, results, resultCount);
            return results;
        }
        finally
        {
            pool.Return(scored, clearArray: true);
        }
    }

    /// <summary>
    /// Min-heap + ArrayPool: best of both optimizations.
    /// </summary>
    [Benchmark]
    public (string Id, float Similarity)[] MinHeap_ArrayPool()
    {
        int k = Math.Min(TopK, _sourceData.Length);
        var pool = ArrayPool<(string Id, float Similarity)>.Shared;
        var heap = pool.Rent(k);
        try
        {
            int heapSize = 0;

            for (int i = 0; i < _sourceData.Length; i++)
            {
                var item = _sourceData[i];

                if (heapSize < k)
                {
                    heap[heapSize] = item;
                    heapSize++;
                    if (heapSize == k)
                        BuildMinHeap(heap, k);
                }
                else if (CompareDescending(item, heap[0]) < 0)
                {
                    heap[0] = item;
                    SiftDown(heap, 0, k);
                }
            }

            // Sort the final heap for deterministic order
            Array.Sort(heap, 0, heapSize, Comparer<(string Id, float Similarity)>.Create(
                (a, b) =>
                {
                    int cmp = b.Similarity.CompareTo(a.Similarity);
                    return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
                }));

            var results = new (string Id, float Similarity)[heapSize];
            Array.Copy(heap, results, heapSize);
            return results;
        }
        finally
        {
            pool.Return(heap, clearArray: true);
        }
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
