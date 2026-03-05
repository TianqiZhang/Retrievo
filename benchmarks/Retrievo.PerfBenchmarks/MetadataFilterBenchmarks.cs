using System.Collections.Frozen;
using BenchmarkDotNet.Attributes;

namespace Retrievo.PerfBenchmarks;

/// <summary>
/// Benchmarks comparing string.Split (allocation-heavy) vs Span-based splitting
/// for metadata filter evaluation, and FrozenDictionary vs Dictionary for
/// read-heavy metadata lookups.
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class MetadataFilterBenchmarks
{
    private string[] _storedValues = null!;
    private string _filterValue = null!;
    private Dictionary<string, string> _metadata = null!;
    private FrozenDictionary<string, string> _frozenMetadata = null!;
    private string[] _lookupKeys = null!;

    [Params(10, 50)]
    public int FieldCount { get; set; }

    [Params(5, 20)]
    public int ArrayElements { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        // Generate pipe-delimited stored values (simulating StringArray fields)
        _storedValues = new string[FieldCount];
        for (int f = 0; f < FieldCount; f++)
        {
            var elements = new string[ArrayElements];
            for (int e = 0; e < ArrayElements; e++)
                elements[e] = $"tag-{rng.Next(1000):D4}";
            _storedValues[f] = string.Join('|', elements);
        }

        // The filter value to search for — pick one that exists
        _filterValue = _storedValues[0].Split('|')[ArrayElements / 2];

        // Generate metadata dictionary for lookup benchmarks
        _metadata = new Dictionary<string, string>(FieldCount, StringComparer.Ordinal);
        for (int i = 0; i < FieldCount; i++)
            _metadata[$"field-{i:D3}"] = $"value-{i:D3}";
        _frozenMetadata = _metadata.ToFrozenDictionary(StringComparer.Ordinal);

        // Keys to look up (mix of hits and misses)
        _lookupKeys = new string[FieldCount * 2];
        for (int i = 0; i < FieldCount; i++)
            _lookupKeys[i] = $"field-{i:D3}"; // hit
        for (int i = FieldCount; i < FieldCount * 2; i++)
            _lookupKeys[i] = $"field-{i:D3}"; // miss
    }

    // ── String Split: string.Split vs Span-based ─────────────────────────

    /// <summary>
    /// Current: string.Split allocates a new string[] on every call.
    /// </summary>
    [Benchmark(Baseline = true)]
    public bool ContainsMatch_StringSplit()
    {
        bool anyFound = false;
        for (int f = 0; f < _storedValues.Length; f++)
        {
            foreach (var segment in _storedValues[f].Split('|', StringSplitOptions.RemoveEmptyEntries))
            {
                if (string.Equals(segment, _filterValue, StringComparison.Ordinal))
                {
                    anyFound = true;
                    break;
                }
            }
        }
        return anyFound;
    }

    /// <summary>
    /// Optimized: MemoryExtensions.Split with Span&lt;Range&gt; — zero-alloc splitting.
    /// </summary>
    [Benchmark]
    public bool ContainsMatch_SpanSplit()
    {
        bool anyFound = false;
        Span<Range> ranges = stackalloc Range[128]; // sufficient for most fields

        for (int f = 0; f < _storedValues.Length; f++)
        {
            var valueSpan = _storedValues[f].AsSpan();
            int count = valueSpan.Split(ranges, '|', StringSplitOptions.RemoveEmptyEntries);

            for (int i = 0; i < count; i++)
            {
                if (valueSpan[ranges[i]].SequenceEqual(_filterValue.AsSpan()))
                {
                    anyFound = true;
                    break;
                }
            }
        }
        return anyFound;
    }

    /// <summary>
    /// Manual scan: walk the string looking for delimiters — no allocation.
    /// </summary>
    [Benchmark]
    public bool ContainsMatch_ManualScan()
    {
        bool anyFound = false;

        for (int f = 0; f < _storedValues.Length; f++)
        {
            var span = _storedValues[f].AsSpan();
            var filter = _filterValue.AsSpan();

            while (span.Length > 0)
            {
                int idx = span.IndexOf('|');
                ReadOnlySpan<char> segment;
                if (idx < 0)
                {
                    segment = span;
                    span = ReadOnlySpan<char>.Empty;
                }
                else
                {
                    segment = span[..idx];
                    span = span[(idx + 1)..];
                }

                if (segment.Length > 0 && segment.SequenceEqual(filter))
                {
                    anyFound = true;
                    break;
                }
            }
        }
        return anyFound;
    }

    // ── Dictionary Lookup: Dictionary vs FrozenDictionary ────────────────

    /// <summary>
    /// Current: Dictionary&lt;string, string&gt;.TryGetValue for metadata lookups.
    /// </summary>
    [Benchmark]
    public int DictionaryLookup_Regular()
    {
        int hits = 0;
        for (int i = 0; i < _lookupKeys.Length; i++)
        {
            if (_metadata.TryGetValue(_lookupKeys[i], out _))
                hits++;
        }
        return hits;
    }

    /// <summary>
    /// FrozenDictionary — optimized for read-heavy workloads (.NET 8).
    /// </summary>
    [Benchmark]
    public int DictionaryLookup_Frozen()
    {
        int hits = 0;
        for (int i = 0; i < _lookupKeys.Length; i++)
        {
            if (_frozenMetadata.TryGetValue(_lookupKeys[i], out _))
                hits++;
        }
        return hits;
    }
}
