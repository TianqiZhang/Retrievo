using System.Diagnostics;
using Retrievo.Abstractions;
using Retrievo.Models;

namespace Retrievo.Benchmarks.Evaluation;

/// <summary>
/// Runs BEIR evaluation: builds index, executes queries, computes metrics.
/// </summary>
public sealed class EvaluationRunner : IDisposable
{
    private readonly IReadOnlyDictionary<string, string> _queries;
    private readonly IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> _qrels;
    private readonly IReadOnlyDictionary<string, float[]>? _embeddings;
    private readonly IHybridSearchIndex _index;
    private readonly bool _disposeIndex;
    private bool _disposed;

    /// <summary>
    /// Initialize a new evaluation runner and build the index once.
    /// </summary>
    /// <param name="documents">Corpus documents used to build the index.</param>
    /// <param name="queries">Query text keyed by query ID.</param>
    /// <param name="qrels">Relevance judgments keyed by query ID.</param>
    /// <param name="embeddings">Optional embedding dictionary keyed by ID (documents and queries).</param>
    public EvaluationRunner(
        IReadOnlyList<Document> documents,
        IReadOnlyDictionary<string, string> queries,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> qrels,
        IReadOnlyDictionary<string, float[]>? embeddings = null)
        : this(
            BuildIndex(documents),
            queries,
            qrels,
            embeddings,
            disposeIndex: true)
    {
    }

    /// <summary>
    /// Initialize a new evaluation runner for an existing index instance.
    /// </summary>
    /// <param name="index">Search index to evaluate.</param>
    /// <param name="queries">Query text keyed by query ID.</param>
    /// <param name="qrels">Relevance judgments keyed by query ID.</param>
    /// <param name="embeddings">Optional embedding dictionary keyed by ID (documents and queries).</param>
    /// <param name="disposeIndex">Whether the runner should dispose the supplied index when disposed.</param>
    public EvaluationRunner(
        IHybridSearchIndex index,
        IReadOnlyDictionary<string, string> queries,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> qrels,
        IReadOnlyDictionary<string, float[]>? embeddings = null,
        bool disposeIndex = false)
    {
        ArgumentNullException.ThrowIfNull(index);
        ArgumentNullException.ThrowIfNull(queries);
        ArgumentNullException.ThrowIfNull(qrels);

        if (index.GetStats().DocumentCount == 0)
            throw new InvalidOperationException("Cannot evaluate with an empty corpus.");

        _index = index;
        _queries = queries;
        _qrels = qrels;
        _embeddings = embeddings;
        _disposeIndex = disposeIndex;
    }

    /// <summary>
    /// Run evaluation with the specified configuration.
    /// </summary>
    /// <param name="config">Evaluation configuration.</param>
    /// <returns>Aggregated evaluation metrics and timing.</returns>
    public EvaluationResult Run(EvaluationConfig config) => RunDetailed(config).Summary;

    /// <summary>
    /// Run evaluation with the specified configuration and capture per-query rankings.
    /// </summary>
    /// <param name="config">Evaluation configuration.</param>
    /// <returns>Aggregated evaluation metrics, timing, and per-query ranked outputs.</returns>
    public EvaluationRun RunDetailed(EvaluationConfig config)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(config);

        if (!config.UseLexical && !config.UseVector)
            throw new ArgumentException("Configuration must enable lexical search, vector search, or both.", nameof(config));

        if (config.TopK <= 0)
            throw new ArgumentOutOfRangeException(nameof(config), "TopK must be greater than zero.");
        if (config.RecallAtK <= 0)
            throw new ArgumentOutOfRangeException(nameof(config), "RecallAtK must be greater than zero.");

        var totalTimer = Stopwatch.StartNew();

        var ndcgTotal = 0d;
        var mapTotal = 0d;
        var recallTotal = 0d;
        var queryTimeTotalMs = 0d;
        var queryCount = 0;
        var queryResults = new Dictionary<string, QueryEvaluationResult>(StringComparer.Ordinal);

        foreach (var (queryId, qrelForQuery) in _qrels)
        {
            if (!_queries.TryGetValue(queryId, out var queryText))
                continue;

            float[]? queryVector = null;
            if (config.UseVector)
            {
                if (_embeddings is null || !_embeddings.TryGetValue(queryId, out queryVector))
                    continue;
            }

            var queryForTopK = new HybridQuery
            {
                Text = config.UseLexical ? queryText : null,
                Vector = config.UseVector ? queryVector : null,
                TopK = config.TopK,
                LexicalK = config.LexicalK,
                VectorK = config.VectorK,
                LexicalWeight = config.LexicalWeight,
                VectorWeight = config.VectorWeight,
                RrfK = config.RrfK,
                TitleBoost = config.TitleBoost
            };

            var responseAtTopK = _index.Search(queryForTopK);
            queryTimeTotalMs += responseAtTopK.QueryTimeMs;

            var rankedAtTopK = responseAtTopK.Results.Select(result => result.Id).ToList();

            var queryForRecall = queryForTopK with { TopK = Math.Max(config.RecallAtK, config.TopK) };
            var responseForRecall = queryForTopK.TopK >= config.RecallAtK
                ? responseAtTopK
                : _index.Search(queryForRecall);

            var rankedForRecall = responseForRecall.Results.Select(result => result.Id).ToList();

            ndcgTotal += NdcgCalculator.ComputeNdcg(rankedAtTopK, qrelForQuery, config.TopK);
            mapTotal += NdcgCalculator.ComputeAveragePrecision(rankedAtTopK, qrelForQuery, config.TopK);
            recallTotal += NdcgCalculator.ComputeRecall(rankedForRecall, qrelForQuery, config.RecallAtK);
            queryResults.Add(queryId, new QueryEvaluationResult
            {
                QueryId = queryId,
                RankedAtTopK = rankedAtTopK,
                RankedAtRecallK = rankedForRecall
            });
            queryCount++;
        }

        totalTimer.Stop();

        if (queryCount == 0)
            throw new InvalidOperationException($"Configuration '{config.Name}' evaluated zero queries.");

        return new EvaluationRun
        {
            Summary = new EvaluationResult
            {
                Name = config.Name,
                NdcgAt10 = ndcgTotal / queryCount,
                MapAt10 = mapTotal / queryCount,
                RecallAt100 = recallTotal / queryCount,
                AvgQueryTimeMs = queryTimeTotalMs / queryCount,
                TotalTimeMs = totalTimer.Elapsed.TotalMilliseconds
            },
            QueryResults = queryResults
        };
    }

    /// <summary>
    /// Dispose the underlying index resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        if (_disposeIndex)
            _index.Dispose();

        _disposed = true;
    }

    private static HybridSearchIndex BuildIndex(IReadOnlyList<Document> documents)
    {
        ArgumentNullException.ThrowIfNull(documents);

        if (documents.Count == 0)
            throw new InvalidOperationException("Cannot evaluate with an empty corpus.");

        return new HybridSearchIndexBuilder()
            .AddDocuments(documents)
            .Build();
    }
}

/// <summary>
/// Configuration for a benchmark run.
/// </summary>
public sealed record EvaluationConfig
{
    /// <summary>
    /// Display name for this configuration.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Whether to include the lexical query text component.
    /// </summary>
    public bool UseLexical { get; init; } = true;

    /// <summary>
    /// Whether to include the vector query component.
    /// </summary>
    public bool UseVector { get; init; }

    /// <summary>
    /// Final result cutoff for nDCG@10 and MAP@10.
    /// </summary>
    public int TopK { get; init; } = 10;

    /// <summary>
    /// Candidate count for lexical retrieval before fusion.
    /// </summary>
    public int LexicalK { get; init; } = 100;

    /// <summary>
    /// Candidate count for vector retrieval before fusion.
    /// </summary>
    public int VectorK { get; init; } = 100;

    /// <summary>
    /// Weight for lexical ranking in fusion.
    /// </summary>
    public float LexicalWeight { get; init; } = 0.5f;

    /// <summary>
    /// Weight for vector ranking in fusion.
    /// </summary>
    public float VectorWeight { get; init; } = 1f;

    /// <summary>
    /// Cutoff for recall metric computation.
    /// </summary>
    public int RecallAtK { get; init; } = 100;

    /// <summary>
    /// The k constant in the RRF formula. Default 20, tuned via BEIR benchmark sweeps.
    /// </summary>
    public int RrfK { get; init; } = 20;

    /// <summary>
    /// Boost multiplier for the title field during lexical search. Default 0.5.
    /// </summary>
    public float TitleBoost { get; init; } = 0.5f;
}

/// <summary>
/// Aggregated metrics for one benchmark configuration.
/// </summary>
public sealed record EvaluationResult
{
    /// <summary>
    /// Display name of this configuration.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Mean nDCG@10 across evaluated queries.
    /// </summary>
    public required double NdcgAt10 { get; init; }

    /// <summary>
    /// Mean MAP@10 across evaluated queries.
    /// </summary>
    public required double MapAt10 { get; init; }

    /// <summary>
    /// Mean Recall@100 across evaluated queries.
    /// </summary>
    public required double RecallAt100 { get; init; }

    /// <summary>
    /// Mean query latency in milliseconds.
    /// </summary>
    public required double AvgQueryTimeMs { get; init; }

    /// <summary>
    /// End-to-end elapsed time in milliseconds.
    /// </summary>
    public required double TotalTimeMs { get; init; }
}

/// <summary>
/// Detailed evaluation output for a single configuration, including per-query rankings.
/// </summary>
public sealed record EvaluationRun
{
    /// <summary>
    /// Aggregated metrics and timing for the configuration.
    /// </summary>
    public required EvaluationResult Summary { get; init; }

    /// <summary>
    /// Per-query ranked results used to compute the aggregate metrics.
    /// </summary>
    public required IReadOnlyDictionary<string, QueryEvaluationResult> QueryResults { get; init; }
}

/// <summary>
/// Ranked retrieval outputs captured for a single query.
/// </summary>
public sealed record QueryEvaluationResult
{
    /// <summary>
    /// Query identifier.
    /// </summary>
    public required string QueryId { get; init; }

    /// <summary>
    /// Ranked document IDs at the main evaluation cutoff.
    /// </summary>
    public required IReadOnlyList<string> RankedAtTopK { get; init; }

    /// <summary>
    /// Ranked document IDs at the recall cutoff.
    /// </summary>
    public required IReadOnlyList<string> RankedAtRecallK { get; init; }
}
