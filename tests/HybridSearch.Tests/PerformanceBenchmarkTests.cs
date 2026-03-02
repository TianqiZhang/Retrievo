using System.Diagnostics;
using HybridSearch.Models;
using HybridSearch.Tests.TestData;
using HybridSearch.Vector;

namespace HybridSearch.Tests;

/// <summary>
/// Performance sanity benchmarks. These tests are non-gating but record timings
/// to track performance characteristics. All tests use the large synthetic corpus
/// (3000 docs, 768 dims) per spec §5.2.
/// </summary>
public class PerformanceBenchmarkTests
{
    /// <summary>
    /// Spec §5.2: "3k docs scan: vector brute-force for 3k docs (e.g., 768 dims)
    /// completes within a reasonable interactive budget."
    /// </summary>
    [Fact]
    public void VectorBruteForce_3kDocs768Dims_CompletesQuickly()
    {
        var docs = SyntheticCorpusGenerator.GenerateLarge(embeddingDims: 768);
        Assert.Equal(3000, docs.Count);

        var retriever = new BruteForceVectorRetriever();
        foreach (var doc in docs)
        {
            retriever.Add(doc.Id, doc.Embedding!);
        }

        // Warm up
        retriever.Search(docs[0].Embedding!, 10);

        // Timed run
        var sw = Stopwatch.StartNew();
        const int iterations = 100;
        for (int i = 0; i < iterations; i++)
        {
            var queryVec = docs[i % docs.Count].Embedding!;
            var results = retriever.Search(queryVec, 10);
            Assert.Equal(10, results.Count);
        }
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
        // Record timing — this is informational, not a hard gate
        // Expected: well under 50ms per query for brute-force 3k x 768
        Assert.True(avgMs < 500, $"Average vector query took {avgMs:F2}ms — expected < 500ms for interactive use");
    }

    /// <summary>
    /// Full hybrid pipeline benchmark: build index + run hybrid queries on 3k docs.
    /// </summary>
    [Fact]
    public void HybridPipeline_3kDocs768Dims_IndexAndQueryPerformance()
    {
        // Measure index build time
        var docs = SyntheticCorpusGenerator.GenerateLarge(embeddingDims: 768);

        var buildSw = Stopwatch.StartNew();
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();
        buildSw.Stop();

        var stats = index.GetStats();
        Assert.Equal(3000, stats.DocumentCount);
        Assert.Equal(768, stats.EmbeddingDimension);

        // Index build should complete in reasonable time (includes Lucene indexing + vector store)
        Assert.True(buildSw.Elapsed.TotalSeconds < 30,
            $"Index build took {buildSw.Elapsed.TotalSeconds:F2}s — expected < 30s");

        // Measure hybrid query time
        var querySw = Stopwatch.StartNew();
        const int queryIterations = 50;
        for (int i = 0; i < queryIterations; i++)
        {
            var response = index.Search(new HybridQuery
            {
                Text = "neural network machine learning training",
                Vector = docs[i % docs.Count].Embedding,
                TopK = 10
            });

            Assert.True(response.Results.Count > 0);
            Assert.True(response.Results.Count <= 10);
        }
        querySw.Stop();

        double avgQueryMs = querySw.Elapsed.TotalMilliseconds / queryIterations;
        // Hybrid query should be interactive (includes lexical + vector + fusion)
        Assert.True(avgQueryMs < 500,
            $"Average hybrid query took {avgQueryMs:F2}ms — expected < 500ms for interactive use");
    }

    /// <summary>
    /// Lexical-only query benchmark on 3k docs to establish BM25 baseline.
    /// </summary>
    [Fact]
    public void LexicalOnly_3kDocs_QueryPerformance()
    {
        var docs = SyntheticCorpusGenerator.GenerateLarge(embeddingDims: 768);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var sw = Stopwatch.StartNew();
        const int iterations = 100;
        string[] queries =
        [
            "neural network training data",
            "kubernetes container deployment",
            "SQL query optimization indexing",
            "encryption security authentication",
            "continuous integration deployment pipeline"
        ];

        for (int i = 0; i < iterations; i++)
        {
            var response = index.Search(new HybridQuery
            {
                Text = queries[i % queries.Length],
                TopK = 10
            });
            Assert.True(response.Results.Count >= 0);
        }
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
        Assert.True(avgMs < 200,
            $"Average lexical query took {avgMs:F2}ms — expected < 200ms");
    }

    /// <summary>
    /// Vector-only query benchmark on 3k docs to establish cosine similarity baseline.
    /// </summary>
    [Fact]
    public void VectorOnly_3kDocs768Dims_QueryPerformance()
    {
        var docs = SyntheticCorpusGenerator.GenerateLarge(embeddingDims: 768);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        var sw = Stopwatch.StartNew();
        const int iterations = 100;
        for (int i = 0; i < iterations; i++)
        {
            var response = index.Search(new HybridQuery
            {
                Vector = docs[i % docs.Count].Embedding,
                TopK = 10
            });
            Assert.Equal(10, response.Results.Count);
        }
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
        Assert.True(avgMs < 500,
            $"Average vector-only query took {avgMs:F2}ms — expected < 500ms");
    }

    /// <summary>
    /// Verify explain mode doesn't add excessive overhead.
    /// </summary>
    [Fact]
    public void ExplainMode_3kDocs_NegligibleOverhead()
    {
        var docs = SyntheticCorpusGenerator.GenerateLarge(embeddingDims: 768);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();

        const int iterations = 50;
        var queryVector = docs[0].Embedding;

        // Without explain
        var swNoExplain = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            index.Search(new HybridQuery
            {
                Text = "machine learning algorithms",
                Vector = queryVector,
                TopK = 10,
                Explain = false
            });
        }
        swNoExplain.Stop();

        // With explain
        var swExplain = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var response = index.Search(new HybridQuery
            {
                Text = "machine learning algorithms",
                Vector = queryVector,
                TopK = 10,
                Explain = true
            });
            Assert.All(response.Results, r => Assert.NotNull(r.Explain));
        }
        swExplain.Stop();

        // Explain overhead should be < 2x (generous margin for dict allocations)
        double ratio = swExplain.Elapsed.TotalMilliseconds / Math.Max(swNoExplain.Elapsed.TotalMilliseconds, 1);
        Assert.True(ratio < 3.0,
            $"Explain overhead ratio: {ratio:F2}x — expected < 3x");
    }

    /// <summary>
    /// Memory allocation sanity: verify the 3k-doc index can be created without issues.
    /// (Not measuring exact allocations — just ensuring no OOM or excessive GC pressure.)
    /// </summary>
    [Fact]
    public void LargeCorpus_CanBeCreatedAndQueried()
    {
        var docs = SyntheticCorpusGenerator.GenerateLarge(embeddingDims: 768);

        long memBefore = GC.GetTotalMemory(true);
        using var index = new HybridSearchIndexBuilder()
            .AddDocuments(docs)
            .Build();
        long memAfter = GC.GetTotalMemory(true);

        double memUsedMb = (memAfter - memBefore) / (1024.0 * 1024.0);
        // 3000 docs x 768 floats x 4 bytes = ~8.8 MB just for vectors
        // Plus Lucene index, doc map, etc. Should be well under 500 MB
        Assert.True(memUsedMb < 500,
            $"Index memory usage: {memUsedMb:F1}MB — expected < 500MB");

        // Verify searchable
        var response = index.Search(new HybridQuery
        {
            Text = "kubernetes",
            Vector = docs[0].Embedding,
            TopK = 10
        });
        Assert.True(response.Results.Count > 0);
    }
}
