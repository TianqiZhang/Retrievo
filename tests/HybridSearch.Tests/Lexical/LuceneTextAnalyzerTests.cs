using HybridSearch.Lexical;

namespace HybridSearch.Tests.Lexical;

public class LuceneTextAnalyzerTests
{
    [Fact]
    public void Analyze_BasicText_ReturnsLowercaseTokens()
    {
        using var analyzer = new LuceneTextAnalyzer();
        var tokens = analyzer.Analyze("body", "Hello World").ToList();

        Assert.Equal(2, tokens.Count);
        Assert.Equal("hello", tokens[0]);
        Assert.Equal("world", tokens[1]);
    }

    [Fact]
    public void Analyze_RemovesStopWords()
    {
        using var analyzer = new LuceneTextAnalyzer();
        // "the", "a", "is" are standard English stop words
        var tokens = analyzer.Analyze("body", "the quick brown fox is a test").ToList();

        Assert.DoesNotContain("the", tokens);
        Assert.DoesNotContain("is", tokens);
        Assert.DoesNotContain("a", tokens);
        Assert.Contains("quick", tokens);
        Assert.Contains("brown", tokens);
        Assert.Contains("fox", tokens);
        Assert.Contains("test", tokens);
    }

    [Fact]
    public void Analyze_Punctuation_StrippedAndTokenized()
    {
        using var analyzer = new LuceneTextAnalyzer();
        var tokens = analyzer.Analyze("body", "hello, world! this is... great.").ToList();

        Assert.Contains("hello", tokens);
        Assert.Contains("world", tokens);
        Assert.Contains("great", tokens);
    }

    [Fact]
    public void Analyze_EmptyString_ReturnsEmpty()
    {
        using var analyzer = new LuceneTextAnalyzer();
        var tokens = analyzer.Analyze("body", "").ToList();

        Assert.Empty(tokens);
    }

    [Fact]
    public void Analyze_NullText_Throws()
    {
        using var analyzer = new LuceneTextAnalyzer();
        Assert.Throws<ArgumentNullException>(() => analyzer.Analyze("body", null!).ToList());
    }

    [Fact]
    public void Analyze_NullFieldName_Throws()
    {
        using var analyzer = new LuceneTextAnalyzer();
        Assert.Throws<ArgumentNullException>(() => analyzer.Analyze(null!, "hello").ToList());
    }

    [Fact]
    public void Dispose_ThenAnalyze_Throws()
    {
        var analyzer = new LuceneTextAnalyzer();
        analyzer.Dispose();

        Assert.Throws<ObjectDisposedException>(() => analyzer.Analyze("body", "test").ToList());
    }

    [Fact]
    public void Analyze_NumericTokens_Preserved()
    {
        using var analyzer = new LuceneTextAnalyzer();
        var tokens = analyzer.Analyze("body", "version 3.14 release 42").ToList();

        Assert.Contains("version", tokens);
        Assert.Contains("3.14", tokens);
        Assert.Contains("releas", tokens); // Porter-stemmed form of "release"
        Assert.Contains("42", tokens);
    }
}
