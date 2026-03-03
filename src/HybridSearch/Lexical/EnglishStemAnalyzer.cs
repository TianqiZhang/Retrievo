using Lucene.Net.Analysis;
using Lucene.Net.Analysis.Core;
using Lucene.Net.Analysis.En;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Analysis.Util;
using Lucene.Net.Util;
using System.IO;

namespace HybridSearch.Lexical;

/// <summary>
/// Custom English analyzer matching Anserini's DefaultEnglishAnalyzer pipeline.
/// Token pipeline: StandardTokenizer → EnglishPossessiveFilter → LowerCaseFilter
/// → StopFilter (Lucene English stop words) → PorterStemFilter.
/// This improves BM25 recall by normalizing word forms (e.g., "infections" → "infect").
/// </summary>
internal sealed class EnglishStemAnalyzer : Analyzer
{
    private readonly LuceneVersion _matchVersion;
    private readonly CharArraySet _stopWords;

    /// <summary>
    /// Creates an EnglishStemAnalyzer with Lucene's default English stop words.
    /// </summary>
    public EnglishStemAnalyzer(LuceneVersion matchVersion)
        : this(matchVersion, EnglishAnalyzer.DefaultStopSet)
    {
    }

    /// <summary>
    /// Creates an EnglishStemAnalyzer with custom stop words.
    /// </summary>
    public EnglishStemAnalyzer(LuceneVersion matchVersion, CharArraySet stopWords)
    {
        _matchVersion = matchVersion;
        _stopWords = stopWords ?? throw new ArgumentNullException(nameof(stopWords));
    }

    /// <summary>
    /// Builds the token pipeline matching Anserini's DefaultEnglishAnalyzer.
    /// </summary>
    protected override TokenStreamComponents CreateComponents(string fieldName, TextReader reader)
    {
        var tokenizer = new StandardTokenizer(_matchVersion, reader);
        TokenStream result = tokenizer;
        result = new EnglishPossessiveFilter(_matchVersion, result);
        result = new LowerCaseFilter(_matchVersion, result);
        result = new StopFilter(_matchVersion, result, _stopWords);
        result = new PorterStemFilter(result);
        return new TokenStreamComponents(tokenizer, result);
    }
}
