using System.Numerics.Tensors;

namespace Retrievo.Vector;

/// <summary>
/// SIMD-accelerated vector math utilities for cosine similarity computation.
/// Uses System.Numerics.Tensors for hardware-accelerated dot products and norms.
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

        return TensorPrimitives.Dot(a, b);
    }

    /// <summary>
    /// Compute the L2 norm (magnitude) of a vector.
    /// </summary>
    public static float L2Norm(ReadOnlySpan<float> v)
    {
        return TensorPrimitives.Norm(v);
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
