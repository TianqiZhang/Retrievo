using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;

namespace Retrievo.PerfBenchmarks;

/// <summary>
/// Benchmarks comparing RRF score accumulation strategies:
/// - Dictionary TryGetValue + set (current)
/// - CollectionsMarshal.GetValueRefOrAddDefault (zero-copy update)
/// And top-K extraction:
/// - LINQ OrderByDescending().ThenBy().Take().ToList() (current)
/// - Min-heap selection
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class RrfFusionBenchmarks
{
    private List<(string Id, int Rank)>[] _rankedLists = null!;
    private Dictionary<string, double> _preAccumulatedScores = null!;

    [Params(100, 1000, 5000)]
    public int DocumentCount { get; set; }

    [Params(10, 50)]
    public int TopK { get; set; }

    private const int RrfK = 20;
    private const int ListCount = 2; // lexical + vector

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        // Generate two ranked lists (simulating lexical and vector results)
        _rankedLists = new List<(string Id, int Rank)>[ListCount];
        var allIds = Enumerable.Range(0, DocumentCount).Select(i => $"doc-{i:D6}").ToArray();

        for (int list = 0; list < ListCount; list++)
        {
            var shuffled = allIds.OrderBy(_ => rng.Next()).ToArray();
            _rankedLists[list] = new List<(string, int)>(DocumentCount);
            for (int rank = 0; rank < DocumentCount; rank++)
            {
                _rankedLists[list].Add((shuffled[rank], rank + 1)); // 1-based rank
            }
        }

        // Pre-accumulated scores for top-K extraction benchmarks
        _preAccumulatedScores = new Dictionary<string, double>(DocumentCount, StringComparer.Ordinal);
        foreach (var id in allIds)
        {
            _preAccumulatedScores[id] = rng.NextDouble();
        }
    }

    // ── Score Accumulation ───────────────────────────────────────────────

    /// <summary>
    /// Current: TryGetValue + indexer set (two hash lookups per insert).
    /// </summary>
    [Benchmark(Baseline = true)]
    public Dictionary<string, double> Accumulate_TryGetValue()
    {
        var scores = new Dictionary<string, double>(StringComparer.Ordinal);

        for (int list = 0; list < ListCount; list++)
        {
            float weight = list == 0 ? 0.5f : 1.0f;
            var items = _rankedLists[list];

            for (int i = 0; i < items.Count; i++)
            {
                var (id, rank) = items[i];
                double contribution = weight * (1.0 / (RrfK + rank));

                if (!scores.TryGetValue(id, out double existing))
                    existing = 0;
                scores[id] = existing + contribution;
            }
        }

        return scores;
    }

    /// <summary>
    /// Optimized: CollectionsMarshal.GetValueRefOrAddDefault (single hash lookup per update).
    /// </summary>
    [Benchmark]
    public Dictionary<string, double> Accumulate_CollectionsMarshal()
    {
        var scores = new Dictionary<string, double>(StringComparer.Ordinal);

        for (int list = 0; list < ListCount; list++)
        {
            float weight = list == 0 ? 0.5f : 1.0f;
            var items = _rankedLists[list];

            for (int i = 0; i < items.Count; i++)
            {
                var (id, rank) = items[i];
                double contribution = weight * (1.0 / (RrfK + rank));

                ref double scoreRef = ref CollectionsMarshal.GetValueRefOrAddDefault(scores, id, out _);
                scoreRef += contribution;
            }
        }

        return scores;
    }

    // ── Top-K Extraction ─────────────────────────────────────────────────

    /// <summary>
    /// Current: LINQ OrderByDescending().ThenBy().Take().ToList().
    /// </summary>
    [Benchmark]
    public List<KeyValuePair<string, double>> TopK_Linq()
    {
        return _preAccumulatedScores
            .OrderByDescending(kvp => kvp.Value)
            .ThenBy(kvp => kvp.Key, StringComparer.Ordinal)
            .Take(TopK)
            .ToList();
    }

    /// <summary>
    /// Min-heap based top-K extraction — O(n log k) vs O(n log n).
    /// </summary>
    [Benchmark]
    public List<KeyValuePair<string, double>> TopK_MinHeap()
    {
        int k = Math.Min(TopK, _preAccumulatedScores.Count);
        var heap = new KeyValuePair<string, double>[k];
        int heapSize = 0;

        foreach (var kvp in _preAccumulatedScores)
        {
            if (heapSize < k)
            {
                heap[heapSize] = kvp;
                heapSize++;
                if (heapSize == k)
                    BuildMinHeap(heap, k);
            }
            else if (CompareDescending(kvp, heap[0]) < 0)
            {
                heap[0] = kvp;
                SiftDown(heap, 0, k);
            }
        }

        // Sort final heap for deterministic output
        Array.Sort(heap, 0, heapSize, Comparer<KeyValuePair<string, double>>.Create(
            (a, b) =>
            {
                int cmp = b.Value.CompareTo(a.Value);
                return cmp != 0 ? cmp : string.Compare(a.Key, b.Key, StringComparison.Ordinal);
            }));

        return new List<KeyValuePair<string, double>>(heap.AsSpan(0, heapSize).ToArray());
    }

    // ── Heap helpers ─────────────────────────────────────────────────────

    private static int CompareDescending(KeyValuePair<string, double> a, KeyValuePair<string, double> b)
    {
        int cmp = b.Value.CompareTo(a.Value);
        return cmp != 0 ? cmp : string.Compare(a.Key, b.Key, StringComparison.Ordinal);
    }

    private static int CompareAscending(KeyValuePair<string, double> a, KeyValuePair<string, double> b)
    {
        int cmp = a.Value.CompareTo(b.Value);
        return cmp != 0 ? cmp : string.Compare(b.Key, a.Key, StringComparison.Ordinal);
    }

    private static void BuildMinHeap(KeyValuePair<string, double>[] heap, int size)
    {
        for (int i = size / 2 - 1; i >= 0; i--)
            SiftDown(heap, i, size);
    }

    private static void SiftDown(KeyValuePair<string, double>[] heap, int i, int size)
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
