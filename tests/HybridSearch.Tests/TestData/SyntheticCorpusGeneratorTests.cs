using HybridSearch.Tests.TestData;
using HybridSearch.Vector;

namespace HybridSearch.Tests.TestData;

public class SyntheticCorpusGeneratorTests
{
    [Fact]
    public void GenerateSmall_Returns5Documents()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();

        Assert.Equal(5, docs.Count);
    }

    [Fact]
    public void GenerateMedium_Returns100Documents()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium();

        Assert.Equal(100, docs.Count);
    }

    [Fact]
    public void GenerateLarge_Returns3000Documents()
    {
        var docs = SyntheticCorpusGenerator.GenerateLarge();

        Assert.Equal(3000, docs.Count);
    }

    [Fact]
    public void AllDocuments_HaveRequiredFields()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium();

        foreach (var doc in docs)
        {
            Assert.NotNull(doc.Id);
            Assert.NotEmpty(doc.Id);
            Assert.NotNull(doc.Title);
            Assert.NotEmpty(doc.Title);
            Assert.NotNull(doc.Body);
            Assert.NotEmpty(doc.Body);
            Assert.NotNull(doc.Embedding);
            Assert.True(doc.Embedding!.Length > 0);
            Assert.NotNull(doc.Metadata);
        }
    }

    [Fact]
    public void DocumentIds_AreUnique()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium();
        var ids = docs.Select(d => d.Id).ToHashSet();

        Assert.Equal(docs.Count, ids.Count);
    }

    [Fact]
    public void Embeddings_AreNormalized()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();

        foreach (var doc in docs)
        {
            float norm = VectorMath.L2Norm(doc.Embedding!);
            Assert.Equal(1.0f, norm, tolerance: 1e-5f);
        }
    }

    [Fact]
    public void Embeddings_HaveCorrectDimensions()
    {
        var smallDocs = SyntheticCorpusGenerator.GenerateSmall(embeddingDims: 128);
        Assert.All(smallDocs, d => Assert.Equal(128, d.Embedding!.Length));

        var largeDocs = SyntheticCorpusGenerator.GenerateLarge(embeddingDims: 768);
        Assert.All(largeDocs, d => Assert.Equal(768, d.Embedding!.Length));
    }

    [Fact]
    public void SameTopic_DocumentsHaveHigherSimilarity()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium(embeddingDims: 128);

        // Documents at indices 0 and 6 share the same primary topic (topic 0, since 6 % 6 == 0)
        var doc0 = docs[0]; // topic 0
        var doc6 = docs[6]; // topic 0
        var doc1 = docs[1]; // topic 1

        float sameSim = VectorMath.DotProduct(doc0.Embedding!, doc6.Embedding!);
        float diffSim = VectorMath.DotProduct(doc0.Embedding!, doc1.Embedding!);

        // Same-topic similarity should be higher than cross-topic
        Assert.True(sameSim > diffSim,
            $"Expected same-topic similarity ({sameSim:F4}) > cross-topic similarity ({diffSim:F4})");
    }

    [Fact]
    public void Generation_IsDeterministic()
    {
        var docs1 = SyntheticCorpusGenerator.GenerateSmall();
        var docs2 = SyntheticCorpusGenerator.GenerateSmall();

        for (int i = 0; i < docs1.Count; i++)
        {
            Assert.Equal(docs1[i].Id, docs2[i].Id);
            Assert.Equal(docs1[i].Title, docs2[i].Title);
            Assert.Equal(docs1[i].Body, docs2[i].Body);
            Assert.Equal(docs1[i].Embedding, docs2[i].Embedding);
        }
    }

    [Fact]
    public void Documents_HaveMultipleSentences()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();

        foreach (var doc in docs)
        {
            // Each document should have at least 4 sentences (3 primary + 1 secondary)
            int sentenceCount = doc.Body.Count(c => c == '.');
            Assert.True(sentenceCount >= 4,
                $"Document '{doc.Id}' has only {sentenceCount} sentences: {doc.Body}");
        }
    }

    [Fact]
    public void Metadata_ContainsTopicAndIndex()
    {
        var docs = SyntheticCorpusGenerator.GenerateSmall();

        foreach (var doc in docs)
        {
            Assert.True(doc.Metadata!.ContainsKey("topic"));
            Assert.True(doc.Metadata!.ContainsKey("index"));
            Assert.True(doc.Metadata!.ContainsKey("source"));
            Assert.Equal("synthetic", doc.Metadata!["source"]);
        }
    }

    [Fact]
    public void Documents_CoverMultipleTopics()
    {
        var docs = SyntheticCorpusGenerator.GenerateMedium();
        var topics = docs.Select(d => d.Metadata!["topic"]).Distinct().ToList();

        // We have 6 topics, medium corpus (100 docs) should cover all
        Assert.True(topics.Count >= 5, $"Expected at least 5 topics, got {topics.Count}: {string.Join(", ", topics)}");
    }

    [Fact]
    public void LargeCorpus_GeneratesWithin5Seconds()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var docs = SyntheticCorpusGenerator.GenerateLarge();
        sw.Stop();

        Assert.Equal(3000, docs.Count);
        Assert.True(sw.ElapsedMilliseconds < 5000,
            $"Large corpus generation took {sw.ElapsedMilliseconds}ms, expected < 5000ms");
    }
}
