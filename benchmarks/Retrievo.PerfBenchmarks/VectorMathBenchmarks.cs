using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Retrievo.Vector;
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
    private float[] _rawVectorA = null!;
    private float[] _rawVectorB = null!;
    private float[] _vectorA = null!;
    private float[] _vectorB = null!;
    private float[] _normalizeBuffer = null!;

    [Params(384, 768, 1536)]
    public int Dimensions { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _rawVectorA = new float[Dimensions];
        _rawVectorB = new float[Dimensions];
        _normalizeBuffer = new float[Dimensions];

        for (int i = 0; i < Dimensions; i++)
        {
            _rawVectorA[i] = (float)(rng.NextDouble() * 2 - 1);
            _rawVectorB[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // Pre-normalize like the real code does
        _vectorA = VectorMath.Normalize(_rawVectorA);
        _vectorB = VectorMath.Normalize(_rawVectorB);
    }

    [IterationSetup]
    public void ResetBuffers()
    {
        _rawVectorA.AsSpan().CopyTo(_normalizeBuffer);
    }

    // ── Dot Product ──────────────────────────────────────────────────────

    /// <summary>
    /// Current implementation: deterministic SIMD accumulation with scalar tail.
    /// </summary>
    [Benchmark(Baseline = true)]
    public float DotProduct_Current()
    {
        return VectorMath.DotProduct(_vectorA, _vectorB);
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
    /// Current implementation: deterministic SIMD accumulation with scalar tail.
    /// </summary>
    [Benchmark]
    public float L2Norm_Current()
    {
        return VectorMath.L2Norm(_vectorA);
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

    // ── Normalize In Place ───────────────────────────────────────────────

    /// <summary>
    /// Current implementation: deterministic SIMD accumulation followed by scalar rescale.
    /// </summary>
    [Benchmark]
    public float NormalizeInPlace_Current()
    {
        return VectorMath.NormalizeInPlace(_normalizeBuffer);
    }

    /// <summary>
    /// TensorPrimitives.Norm for the norm calculation, with the same scalar rescale.
    /// </summary>
    [Benchmark]
    public float NormalizeInPlace_TensorPrimitives()
    {
        return TensorNormalizeInPlace(_normalizeBuffer);
    }

    /// <summary>
    /// Scalar-only normalization for baseline comparison.
    /// </summary>
    [Benchmark]
    public float NormalizeInPlace_Scalar()
    {
        return ScalarNormalizeInPlace(_normalizeBuffer);
    }

    // ── Cosine Similarity (end-to-end) ───────────────────────────────────

    /// <summary>
    /// Current approach: pre-normalize both vectors, then dot product.
    /// This benchmarks the combined normalize+dot path for un-normalized input.
    /// </summary>
    [Benchmark]
    public float CosineSimilarity_NormalizeThenDot()
    {
        var normA = VectorMath.Normalize(_rawVectorA);
        var normB = VectorMath.Normalize(_rawVectorB);
        return VectorMath.DotProduct(normA, normB);
    }

    /// <summary>
    /// TensorPrimitives.CosineSimilarity — single-pass, no intermediate allocations.
    /// </summary>
    [Benchmark]
    public float CosineSimilarity_TensorPrimitives()
    {
        return TensorPrimitives.CosineSimilarity(_vectorA.AsSpan(), _vectorB.AsSpan());
    }

    // ── Comparison helpers ────────────────────────────────────────────────

    private static float ScalarDotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
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

    private static float TensorNormalizeInPlace(Span<float> v)
    {
        float norm = TensorPrimitives.Norm(v);
        if (norm == 0f)
            return 0f;

        float inv = 1f / norm;
        for (int i = 0; i < v.Length; i++)
            v[i] *= inv;

        return norm;
    }

    private static float ScalarNormalizeInPlace(Span<float> v)
    {
        float sum = 0f;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];

        float norm = MathF.Sqrt(sum);
        if (norm == 0f)
            return 0f;

        float inv = 1f / norm;
        for (int i = 0; i < v.Length; i++)
            v[i] *= inv;

        return norm;
    }
}
