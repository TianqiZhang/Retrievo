using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using SysVector = System.Numerics.Vector;
using SysVectorFloat = System.Numerics.Vector<float>;

namespace Retrievo.PerfBenchmarks;

/// <summary>
/// Benchmarks comparing vector math approaches for dot product, L2 norm, normalization,
/// and cosine similarity computation.
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class VectorMathBenchmarks
{
    private float[] _vectorA = null!;
    private float[] _vectorB = null!;

    [Params(384, 768, 1536)]
    public int Dimensions { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _vectorA = new float[Dimensions];
        _vectorB = new float[Dimensions];

        for (int i = 0; i < Dimensions; i++)
        {
            _vectorA[i] = (float)(rng.NextDouble() * 2 - 1);
            _vectorB[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // Pre-normalize like the real code does
        NormalizeInPlace(_vectorA);
        NormalizeInPlace(_vectorB);
    }

    // ── Dot Product ──────────────────────────────────────────────────────

    /// <summary>
    /// Current implementation: System.Numerics.Vector&lt;float&gt; SIMD dot product.
    /// </summary>
    [Benchmark(Baseline = true)]
    public float DotProduct_Current()
    {
        return CurrentSimdDotProduct(_vectorA, _vectorB);
    }

    /// <summary>
    /// .NET 8 TensorPrimitives.Dot — hardware-optimized, auto-vectorized.
    /// </summary>
    [Benchmark]
    public float DotProduct_TensorPrimitives()
    {
        return TensorPrimitives.Dot(_vectorA.AsSpan(), _vectorB.AsSpan());
    }

    /// <summary>
    /// Scalar-only dot product for baseline comparison.
    /// </summary>
    [Benchmark]
    public float DotProduct_Scalar()
    {
        return ScalarDotProduct(_vectorA, _vectorB);
    }

    // ── L2 Norm ──────────────────────────────────────────────────────────

    /// <summary>
    /// Current implementation: scalar L2 norm (no SIMD).
    /// </summary>
    [Benchmark]
    public float L2Norm_Current()
    {
        return CurrentScalarL2Norm(_vectorA);
    }

    /// <summary>
    /// TensorPrimitives.Norm — SIMD-accelerated L2 norm.
    /// </summary>
    [Benchmark]
    public float L2Norm_TensorPrimitives()
    {
        return TensorPrimitives.Norm(_vectorA.AsSpan());
    }

    /// <summary>
    /// SIMD L2 norm using System.Numerics.Vector&lt;float&gt;.
    /// </summary>
    [Benchmark]
    public float L2Norm_SimdVector()
    {
        return SimdL2Norm(_vectorA);
    }

    // ── Cosine Similarity (end-to-end) ───────────────────────────────────

    /// <summary>
    /// Current approach: pre-normalize both vectors, then dot product.
    /// This benchmarks the combined normalize+dot path for un-normalized input.
    /// </summary>
    [Benchmark]
    public float CosineSimilarity_NormalizeThenDot()
    {
        // Simulate what happens in the real code: normalize copy + dot
        var normA = new float[Dimensions];
        var normB = new float[Dimensions];
        _vectorA.AsSpan().CopyTo(normA);
        _vectorB.AsSpan().CopyTo(normB);
        NormalizeInPlace(normA);
        NormalizeInPlace(normB);
        return CurrentSimdDotProduct(normA, normB);
    }

    /// <summary>
    /// TensorPrimitives.CosineSimilarity — single-pass, no intermediate allocations.
    /// </summary>
    [Benchmark]
    public float CosineSimilarity_TensorPrimitives()
    {
        return TensorPrimitives.CosineSimilarity(_vectorA.AsSpan(), _vectorB.AsSpan());
    }

    // ── Implementation helpers (inlined for benchmark isolation) ─────────

    private static float CurrentSimdDotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        int i = 0;

        if (SysVector.IsHardwareAccelerated && a.Length >= SysVectorFloat.Count)
        {
            var spanA = MemoryMarshal.Cast<float, SysVectorFloat>(a);
            var spanB = MemoryMarshal.Cast<float, SysVectorFloat>(b);

            var vSum = SysVectorFloat.Zero;
            for (int v = 0; v < spanA.Length; v++)
            {
                vSum += spanA[v] * spanB[v];
            }

            sum = SysVector.Sum(vSum);
            i = spanA.Length * SysVectorFloat.Count;
        }

        for (; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    private static float ScalarDotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static float CurrentScalarL2Norm(ReadOnlySpan<float> v)
    {
        float sum = 0f;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
        return MathF.Sqrt(sum);
    }

    private static float SimdL2Norm(ReadOnlySpan<float> v)
    {
        float sum = 0f;
        int i = 0;

        if (SysVector.IsHardwareAccelerated && v.Length >= SysVectorFloat.Count)
        {
            var span = MemoryMarshal.Cast<float, SysVectorFloat>(v);

            var vSum = SysVectorFloat.Zero;
            for (int vi = 0; vi < span.Length; vi++)
            {
                vSum += span[vi] * span[vi];
            }

            sum = SysVector.Sum(vSum);
            i = span.Length * SysVectorFloat.Count;
        }

        for (; i < v.Length; i++)
            sum += v[i] * v[i];

        return MathF.Sqrt(sum);
    }

    private static void NormalizeInPlace(Span<float> v)
    {
        float sum = 0f;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
        float norm = MathF.Sqrt(sum);
        if (norm == 0f) return;
        float inv = 1f / norm;
        for (int i = 0; i < v.Length; i++)
            v[i] *= inv;
    }
}
