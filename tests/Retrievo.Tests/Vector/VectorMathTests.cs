using Retrievo.Vector;

namespace Retrievo.Tests.Vector;

public class VectorMathTests
{
    [Fact]
    public void DotProduct_IdenticalNormalizedVectors_ReturnsOne()
    {
        var v = VectorMath.Normalize(new float[] { 1f, 2f, 3f });
        float dot = VectorMath.DotProduct(v, v);
        Assert.Equal(1.0f, dot, tolerance: 1e-6f);
    }

    [Fact]
    public void DotProduct_OrthogonalVectors_ReturnsZero()
    {
        var a = new float[] { 1f, 0f, 0f };
        var b = new float[] { 0f, 1f, 0f };
        float dot = VectorMath.DotProduct(a, b);
        Assert.Equal(0f, dot, tolerance: 1e-6f);
    }

    [Fact]
    public void DotProduct_OppositeVectors_ReturnsNegativeOne()
    {
        var a = VectorMath.Normalize(new float[] { 1f, 0f, 0f });
        var b = VectorMath.Normalize(new float[] { -1f, 0f, 0f });
        float dot = VectorMath.DotProduct(a, b);
        Assert.Equal(-1.0f, dot, tolerance: 1e-6f);
    }

    [Fact]
    public void DotProduct_MismatchedDimensions_Throws()
    {
        var a = new float[] { 1f, 2f };
        var b = new float[] { 1f, 2f, 3f };
        Assert.Throws<ArgumentException>(() => VectorMath.DotProduct(a, b));
    }

    [Fact]
    public void Normalize_ProducesUnitVector()
    {
        var v = VectorMath.Normalize(new float[] { 3f, 4f });
        float norm = VectorMath.L2Norm(v);
        Assert.Equal(1.0f, norm, tolerance: 1e-6f);
    }

    [Fact]
    public void Normalize_ZeroVector_RemainsZero()
    {
        var v = new float[] { 0f, 0f, 0f };
        var result = VectorMath.Normalize(v);
        Assert.All(result, val => Assert.Equal(0f, val));
    }

    [Fact]
    public void DotProduct_LargeVector_SimdPath()
    {
        // Use a vector large enough to exercise the SIMD path (>= Vector<float>.Count)
        int dims = 768;
        var a = new float[dims];
        var b = new float[dims];

        // Create two identical vectors
        for (int i = 0; i < dims; i++)
        {
            a[i] = (float)Math.Sin(i);
            b[i] = (float)Math.Sin(i);
        }

        var aN = VectorMath.Normalize(a);
        var bN = VectorMath.Normalize(b);

        float dot = VectorMath.DotProduct(aN, bN);
        Assert.Equal(1.0f, dot, tolerance: 1e-5f);
    }

    [Fact]
    public void DotProduct_SameValuesWithDifferentSpanAlignment_ReturnsSameBits()
    {
        const int dims = 1536;
        var left = new float[dims];
        var right = new float[dims];

        for (var i = 0; i < dims; i++)
        {
            left[i] = (float)Math.Sin(i * 0.17);
            right[i] = (float)Math.Cos(i * 0.11);
        }

        var direct = VectorMath.DotProduct(left, right);

        var shiftedLeft = new float[dims + 1];
        var shiftedRight = new float[dims + 2];
        Array.Copy(left, 0, shiftedLeft, 1, dims);
        Array.Copy(right, 0, shiftedRight, 2, dims);

        var shifted = VectorMath.DotProduct(
            shiftedLeft.AsSpan(1, dims),
            shiftedRight.AsSpan(2, dims));

        Assert.Equal(
            BitConverter.SingleToInt32Bits(direct),
            BitConverter.SingleToInt32Bits(shifted));
    }
}
