using HybridSearch.Vector;

namespace HybridSearch.Tests.Vector;

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
}
