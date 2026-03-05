using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;

namespace Retrievo.PerfBenchmarks;

/// <summary>
/// Benchmarks comparing validation strategies for checking float[] arrays
/// contain only finite values (no NaN/Infinity).
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class VectorValidationBenchmarks
{
    private float[] _validVector = null!;

    [Params(384, 768, 1536)]
    public int Dimensions { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _validVector = new float[Dimensions];
        for (int i = 0; i < Dimensions; i++)
            _validVector[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    /// <summary>
    /// Current: scalar loop checking float.IsNaN || float.IsInfinity per element.
    /// </summary>
    [Benchmark(Baseline = true)]
    public bool Validate_ScalarLoop()
    {
        for (int i = 0; i < _validVector.Length; i++)
        {
            float value = _validVector[i];
            if (float.IsNaN(value) || float.IsInfinity(value))
                return false;
        }
        return true;
    }

    /// <summary>
    /// Optimized: float.IsFinite per element (single comparison vs two).
    /// </summary>
    [Benchmark]
    public bool Validate_IsFiniteLoop()
    {
        for (int i = 0; i < _validVector.Length; i++)
        {
            if (!float.IsFinite(_validVector[i]))
                return false;
        }
        return true;
    }

    /// <summary>
    /// SIMD: compute dot product with self — NaN/Inf propagates, then check if result is finite.
    /// Uses TensorPrimitives.Dot which is SIMD-accelerated.
    /// </summary>
    [Benchmark]
    public bool Validate_DotSelfCheck()
    {
        ReadOnlySpan<float> span = _validVector;
        float dot = TensorPrimitives.Dot(span, span);
        return float.IsFinite(dot);
    }
}
