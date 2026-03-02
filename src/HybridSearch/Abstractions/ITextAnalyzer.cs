namespace HybridSearch.Abstractions;

/// <summary>
/// Provides text analysis (tokenization, normalization) for lexical indexing and querying.
/// Abstracts the underlying analyzer implementation to allow swapping (e.g., Lucene.NET analyzer
/// today, custom implementation later).
/// </summary>
public interface ITextAnalyzer : IDisposable
{
    /// <summary>
    /// Tokenize and normalize the input text into a sequence of terms.
    /// </summary>
    /// <param name="fieldName">The field name being analyzed (for field-specific analysis).</param>
    /// <param name="text">The raw text to analyze.</param>
    /// <returns>An enumerable of normalized tokens.</returns>
    IEnumerable<string> Analyze(string fieldName, string text);
}
