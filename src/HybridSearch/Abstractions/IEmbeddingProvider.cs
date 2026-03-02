namespace HybridSearch.Abstractions;

/// <summary>
/// Provides text embedding via an external model (e.g., Azure OpenAI, ONNX).
/// All methods are async because embedding typically involves I/O (API calls or model inference).
/// </summary>
public interface IEmbeddingProvider
{
    /// <summary>
    /// Embed a single text string into a dense vector.
    /// </summary>
    /// <param name="text">The text to embed.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The embedding vector.</returns>
    Task<float[]> EmbedAsync(string text, CancellationToken ct = default);

    /// <summary>
    /// Embed a batch of text strings into dense vectors.
    /// Implementations should optimize for batch API calls where possible.
    /// </summary>
    /// <param name="texts">The texts to embed.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>An array of embedding vectors, one per input text, in the same order.</returns>
    Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default);

    /// <summary>
    /// The dimensionality of the embedding vectors produced by this provider.
    /// </summary>
    int Dimensions { get; }
}
