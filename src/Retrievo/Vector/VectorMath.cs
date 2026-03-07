using System.Numerics;

namespace Retrievo.Vector;

/// <summary>
/// SIMD-accelerated vector math utilities for cosine similarity computation.
/// Uses deterministic accumulation so rankings do not depend on span alignment.
/// </summary>
internal static class VectorMath
{
    /// <summary>
    /// Compute the dot product of two float arrays of equal length.
    /// Both arrays must be pre-normalized for the result to equal cosine similarity.
    /// </summary>
    public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Vector dimensions must match: {a.Length} vs {b.Length}");

        return (float)DotProductDeterministic(a, b);
    }

    /// <summary>
    /// Compute the L2 norm (magnitude) of a vector.
    /// </summary>
    public static float L2Norm(ReadOnlySpan<float> v)
    {
        return (float)Math.Sqrt(SumSquaresDeterministic(v));
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

    private static double DotProductDeterministic(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = 0d;
        var width = Vector<float>.Count;
        var i = 0;
        var vectorSum = Vector<float>.Zero;

        for (; i <= a.Length - width; i += width)
        {
            var va = new Vector<float>(a.Slice(i, width));
            var vb = new Vector<float>(b.Slice(i, width));
            vectorSum += va * vb;
        }

        for (var lane = 0; lane < Vector<float>.Count; lane++)
            sum += vectorSum[lane];

        for (; i < a.Length; i++)
            sum += (double)a[i] * b[i];

        return sum;
    }

    private static double SumSquaresDeterministic(ReadOnlySpan<float> v)
    {
        var sum = 0d;
        var width = Vector<float>.Count;
        var i = 0;
        var vectorSum = Vector<float>.Zero;

        for (; i <= v.Length - width; i += width)
        {
            var chunk = new Vector<float>(v.Slice(i, width));
            vectorSum += chunk * chunk;
        }

        for (var lane = 0; lane < Vector<float>.Count; lane++)
            sum += vectorSum[lane];

        for (; i < v.Length; i++)
            sum += (double)v[i] * v[i];

        return sum;
    }
}
