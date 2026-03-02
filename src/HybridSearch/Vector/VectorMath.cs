using System.Numerics;
using System.Runtime.InteropServices;

namespace HybridSearch.Vector;

/// <summary>
/// SIMD-accelerated vector math utilities for cosine similarity computation.
/// Uses System.Numerics.Vector&lt;float&gt; for hardware-accelerated dot products.
/// </summary>
internal static class VectorMath
{
    /// <summary>
    /// Compute the dot product of two float arrays of equal length using SIMD where possible.
    /// Both arrays must be pre-normalized for the result to equal cosine similarity.
    /// </summary>
    public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Vector dimensions must match: {a.Length} vs {b.Length}");

        float sum = 0f;
        int i = 0;

        // SIMD path
        if (System.Numerics.Vector.IsHardwareAccelerated && a.Length >= System.Numerics.Vector<float>.Count)
        {
            var spanA = MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(a);
            var spanB = MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(b);

            var vSum = System.Numerics.Vector<float>.Zero;
            for (int v = 0; v < spanA.Length; v++)
            {
                vSum += spanA[v] * spanB[v];
            }

            sum = System.Numerics.Vector.Sum(vSum);
            i = spanA.Length * System.Numerics.Vector<float>.Count;
        }

        // Scalar remainder
        for (; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>
    /// Compute the L2 norm (magnitude) of a vector.
    /// </summary>
    public static float L2Norm(ReadOnlySpan<float> v)
    {
        float sum = 0f;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
        return MathF.Sqrt(sum);
    }

    /// <summary>
    /// Normalize a vector in-place to unit length. Returns the original norm.
    /// If the vector is zero-length, it remains unchanged and 0 is returned.
    /// </summary>
    public static float NormalizeInPlace(Span<float> v)
    {
        float norm = L2Norm(v);
        if (norm == 0f)
            return 0f;

        float inv = 1f / norm;
        for (int i = 0; i < v.Length; i++)
            v[i] *= inv;

        return norm;
    }

    /// <summary>
    /// Return a new normalized copy of the input vector.
    /// </summary>
    public static float[] Normalize(ReadOnlySpan<float> v)
    {
        var result = new float[v.Length];
        v.CopyTo(result);
        NormalizeInPlace(result);
        return result;
    }
}
