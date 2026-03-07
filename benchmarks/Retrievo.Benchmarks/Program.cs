using System.Diagnostics;
using Retrievo;
using Retrievo.Benchmarks.Data;
using Retrievo.Benchmarks.Embeddings;
using Retrievo.Benchmarks.Evaluation;
using Retrievo.Models;

var exitCode = await RunAsync(args).ConfigureAwait(false);
return exitCode;

static async Task<int> RunAsync(string[] args)
{
    try
    {
        var options = ParseArguments(args);

        var dataDir = Path.GetFullPath(options.DataDir);
        var datasetDir = await BeirDatasetDownloader.EnsureDataAsync(options.Dataset, dataDir).ConfigureAwait(false);

        var corpusPath = Path.Combine(datasetDir, "corpus.jsonl");
        var queriesPath = Path.Combine(datasetDir, "queries.jsonl");
        var qrelsPath = Path.Combine(datasetDir, "qrels", "test.tsv");

        Console.Error.WriteLine("Loading BEIR files...");
        var corpus = await BeirDataLoader.LoadCorpusAsync(corpusPath).ConfigureAwait(false);
        var queries = await BeirDataLoader.LoadQueriesAsync(queriesPath).ConfigureAwait(false);
        var qrels = await BeirDataLoader.LoadQrelsAsync(qrelsPath).ConfigureAwait(false);

        IReadOnlyDictionary<string, float[]>? embeddings = null;
        if (!string.IsNullOrWhiteSpace(options.EmbeddingsPath))
        {
            var embeddingsPath = Path.GetFullPath(options.EmbeddingsPath);
            if (!EmbeddingCache.Exists(embeddingsPath))
            {
                Console.Error.WriteLine($"Embedding cache not found: {embeddingsPath}");
                Console.Error.WriteLine("Continuing with lexical-only evaluation.");
            }
            else
            {
                Console.Error.WriteLine($"Loading embeddings cache: {embeddingsPath}");
                embeddings = await EmbeddingCache.LoadAsync(embeddingsPath).ConfigureAwait(false);
            }
        }

        var documents = BuildDocuments(corpus, embeddings);
        var hasVectorBenchmarks = embeddings is not null && HasQueryEmbeddings(qrels, embeddings);

        if (embeddings is not null && !hasVectorBenchmarks)
            Console.Error.WriteLine("No query embeddings found in cache. Skipping vector-only and hybrid modes.");

        if (options.VerifySnapshotRoundTrip)
        {
            return await RunSnapshotRoundTripVerificationAsync(
                    options,
                    documents,
                    queries,
                    qrels,
                    embeddings,
                    corpus.Count,
                    qrels.Count,
                    hasVectorBenchmarks)
                .ConfigureAwait(false);
        }

        if (options.Sweep)
        {
            using var runner = new EvaluationRunner(documents, queries, qrels, embeddings);
            RunSweep(runner, options.Dataset, corpus.Count, qrels.Count, hasVectorBenchmarks);
        }
        else
        {
            using var runner = new EvaluationRunner(documents, queries, qrels, embeddings);
            RunStandard(runner, options.Dataset, corpus.Count, qrels.Count, hasVectorBenchmarks);
        }

        return 0;
    }
    catch (ArgumentException ex)
    {
        Console.Error.WriteLine($"Invalid argument: {ex.Message}");
        return 1;
    }
    catch (FileNotFoundException ex)
    {
        Console.Error.WriteLine($"File not found: {ex.Message}");
        return 1;
    }
    catch (InvalidDataException ex)
    {
        Console.Error.WriteLine($"Data error: {ex.Message}");
        return 1;
    }
    catch (DirectoryNotFoundException ex)
    {
        Console.Error.WriteLine($"Directory not found: {ex.Message}");
        return 1;
    }
    catch (HttpRequestException ex)
    {
        Console.Error.WriteLine($"Download failed: {ex.Message}");
        return 1;
    }
}

static void RunStandard(
    EvaluationRunner runner,
    string datasetId,
    int corpusCount,
    int queryCount,
    bool hasVectorBenchmarks)
{
    var configs = CreateStandardConfigs(hasVectorBenchmarks);
    var results = RunConfigs(runner, configs);
    PrintResults(datasetId, corpusCount, queryCount, results);
}

static void RunSweep(
    EvaluationRunner runner,
    string datasetId,
    int corpusCount,
    int queryCount,
    bool hasVectorBenchmarks)
{
    var displayName = BeirDatasetDownloader.KnownDatasets.TryGetValue(datasetId, out var info)
        ? info.DisplayName
        : datasetId;

    Console.Error.WriteLine($"Starting parameter sweep for {displayName}...");
    Console.Error.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries");
    Console.Error.WriteLine($"Embeddings: {(hasVectorBenchmarks ? "available (hybrid sweep)" : "not available (lexical-only sweep)")}");

    var configs = CreateSweepConfigs(hasVectorBenchmarks);

    Console.Error.WriteLine($"Running {configs.Count} configurations...");
    Console.Error.WriteLine();

    var timer = Stopwatch.StartNew();
    var results = RunConfigs(runner, configs, showProgress: true);
    timer.Stop();

    // Sort by nDCG@10 descending
    results.Sort((a, b) => b.NdcgAt10.CompareTo(a.NdcgAt10));

    Console.WriteLine($"{displayName} Parameter Sweep (BEIR)");
    Console.WriteLine(new string('=', displayName.Length + 22));
    Console.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries | {configs.Count} configs | {timer.Elapsed.TotalSeconds:F1}s total");
    Console.WriteLine();
    Console.WriteLine($"{"Config",-38} {"nDCG@10",8} {"MAP@10",8} {"R@100",10} {"Avg ms",8}");
    Console.WriteLine(new string('-', 78));

    foreach (var result in results)
    {
        Console.WriteLine(
            $"{result.Name,-38} {result.NdcgAt10,8:F5} {result.MapAt10,8:F5} {result.RecallAt100,10:F5} {result.AvgQueryTimeMs,8:F1}");
    }

    Console.WriteLine();
    Console.WriteLine($"Best: {results[0].Name} -> nDCG@10 = {results[0].NdcgAt10:F5}");
}

static async Task<int> RunSnapshotRoundTripVerificationAsync(
    ParsedOptions options,
    IReadOnlyList<Document> documents,
    IReadOnlyDictionary<string, string> queries,
    IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> qrels,
    IReadOnlyDictionary<string, float[]>? embeddings,
    int corpusCount,
    int queryCount,
    bool hasVectorBenchmarks)
{
    ArgumentNullException.ThrowIfNull(options);
    ArgumentNullException.ThrowIfNull(documents);
    ArgumentNullException.ThrowIfNull(queries);
    ArgumentNullException.ThrowIfNull(qrels);

    var configs = options.Sweep
        ? CreateSweepConfigs(hasVectorBenchmarks)
        : CreateStandardConfigs(hasVectorBenchmarks);

    var displayName = BeirDatasetDownloader.KnownDatasets.TryGetValue(options.Dataset, out var info)
        ? info.DisplayName
        : options.Dataset;

    var snapshotPath = Path.Combine(
        Path.GetTempPath(),
        $"retrievo-{options.Dataset}-snapshot-{Guid.NewGuid():N}.json");

    Console.Error.WriteLine($"Starting snapshot round-trip verification for {displayName}...");
    Console.Error.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries | {configs.Count} configs");
    Console.Error.WriteLine($"Mode: {(options.Sweep ? "parameter sweep" : "standard benchmark")}");
    Console.Error.WriteLine($"Temporary snapshot: {snapshotPath}");

    try
    {
        Console.Error.WriteLine("Building original index...");
        var originalIndex = new HybridSearchIndexBuilder()
            .AddDocuments(documents)
            .Build();

        Console.Error.WriteLine("Exporting snapshot...");
        await originalIndex.ExportSnapshotAsync(snapshotPath).ConfigureAwait(false);

        Console.Error.WriteLine("Running baseline evaluation...");
        using var originalRunner = new EvaluationRunner(originalIndex, queries, qrels, embeddings, disposeIndex: true);
        var originalRuns = RunDetailedConfigs(originalRunner, configs, showProgress: options.Sweep, progressLabel: "baseline");

        Console.Error.WriteLine("Importing snapshot...");
        var restoredIndex = await HybridSearchIndex.ImportSnapshotAsync(snapshotPath).ConfigureAwait(false);

        Console.Error.WriteLine("Running restored evaluation...");
        using var restoredRunner = new EvaluationRunner(restoredIndex, queries, qrels, embeddings, disposeIndex: true);
        var restoredRuns = RunDetailedConfigs(restoredRunner, configs, showProgress: options.Sweep, progressLabel: "snapshot");

        var statsMismatch = CompareIndexStats(originalIndex.GetStats(), restoredIndex.GetStats());
        var mismatch = statsMismatch ?? CompareEvaluationRuns(originalRuns, restoredRuns);

        PrintSnapshotVerificationReport(displayName, corpusCount, queryCount, originalRuns, restoredRuns, mismatch);
        return mismatch is null ? 0 : 1;
    }
    finally
    {
        if (File.Exists(snapshotPath))
            File.Delete(snapshotPath);
    }
}

static List<EvaluationResult> RunConfigs(
    EvaluationRunner runner,
    IReadOnlyList<EvaluationConfig> configs,
    bool showProgress = false)
{
    ArgumentNullException.ThrowIfNull(runner);
    ArgumentNullException.ThrowIfNull(configs);

    var results = new List<EvaluationResult>(configs.Count);
    var timer = showProgress ? Stopwatch.StartNew() : null;

    for (var i = 0; i < configs.Count; i++)
    {
        results.Add(runner.Run(configs[i]));

        if (showProgress && ((i + 1) % 50 == 0 || i + 1 == configs.Count))
            Console.Error.WriteLine($"  [{i + 1}/{configs.Count}] {timer!.Elapsed.TotalSeconds:F1}s");
    }

    return results;
}

static List<EvaluationRun> RunDetailedConfigs(
    EvaluationRunner runner,
    IReadOnlyList<EvaluationConfig> configs,
    bool showProgress,
    string progressLabel)
{
    ArgumentNullException.ThrowIfNull(runner);
    ArgumentNullException.ThrowIfNull(configs);
    ArgumentNullException.ThrowIfNull(progressLabel);

    var runs = new List<EvaluationRun>(configs.Count);
    var timer = showProgress ? Stopwatch.StartNew() : null;

    for (var i = 0; i < configs.Count; i++)
    {
        runs.Add(runner.RunDetailed(configs[i]));

        if (showProgress && ((i + 1) % 50 == 0 || i + 1 == configs.Count))
            Console.Error.WriteLine($"  {progressLabel} [{i + 1}/{configs.Count}] {timer!.Elapsed.TotalSeconds:F1}s");
    }

    return runs;
}

static SnapshotMismatch? CompareIndexStats(IndexStats original, IndexStats restored)
{
    if (original.DocumentCount != restored.DocumentCount)
    {
        return new SnapshotMismatch
        {
            Message = $"Document count mismatch: original={original.DocumentCount}, snapshot={restored.DocumentCount}."
        };
    }

    if (original.EmbeddingDimension != restored.EmbeddingDimension)
    {
        return new SnapshotMismatch
        {
            Message = $"Embedding dimension mismatch: original={original.EmbeddingDimension?.ToString() ?? "null"}, snapshot={restored.EmbeddingDimension?.ToString() ?? "null"}."
        };
    }

    return null;
}

static SnapshotMismatch? CompareEvaluationRuns(
    IReadOnlyList<EvaluationRun> originalRuns,
    IReadOnlyList<EvaluationRun> restoredRuns)
{
    ArgumentNullException.ThrowIfNull(originalRuns);
    ArgumentNullException.ThrowIfNull(restoredRuns);

    if (originalRuns.Count != restoredRuns.Count)
    {
        return new SnapshotMismatch
        {
            Message = $"Configuration count mismatch: original={originalRuns.Count}, snapshot={restoredRuns.Count}."
        };
    }

    for (var i = 0; i < originalRuns.Count; i++)
    {
        var originalRun = originalRuns[i];
        var restoredRun = restoredRuns[i];
        var originalSummary = originalRun.Summary;
        var restoredSummary = restoredRun.Summary;

        if (!string.Equals(originalSummary.Name, restoredSummary.Name, StringComparison.Ordinal))
        {
            return new SnapshotMismatch
            {
                Message = $"Configuration order mismatch at position {i + 1}: original='{originalSummary.Name}', snapshot='{restoredSummary.Name}'."
            };
        }

        if (originalRun.QueryResults.Count != restoredRun.QueryResults.Count)
        {
            return new SnapshotMismatch
            {
                Message = $"Evaluated query count mismatch for '{originalSummary.Name}': original={originalRun.QueryResults.Count}, snapshot={restoredRun.QueryResults.Count}.",
                ConfigurationName = originalSummary.Name
            };
        }

        foreach (var (queryId, originalQueryResult) in originalRun.QueryResults)
        {
            if (!restoredRun.QueryResults.TryGetValue(queryId, out var restoredQueryResult))
            {
                return new SnapshotMismatch
                {
                    Message = $"Snapshot run is missing query '{queryId}' for configuration '{originalSummary.Name}'.",
                    ConfigurationName = originalSummary.Name,
                    QueryId = queryId
                };
            }

            if (!originalQueryResult.RankedAtTopK.SequenceEqual(restoredQueryResult.RankedAtTopK, StringComparer.Ordinal))
            {
                var differenceIndex = FindFirstDifference(originalQueryResult.RankedAtTopK, restoredQueryResult.RankedAtTopK);
                return new SnapshotMismatch
                {
                    Message = $"TopK ranking mismatch for configuration '{originalSummary.Name}', query '{queryId}'.",
                    ConfigurationName = originalSummary.Name,
                    QueryId = queryId,
                    RankingKind = "TopK",
                    DifferenceIndex = differenceIndex,
                    OriginalRanking = originalQueryResult.RankedAtTopK,
                    RestoredRanking = restoredQueryResult.RankedAtTopK
                };
            }

            if (!originalQueryResult.RankedAtRecallK.SequenceEqual(restoredQueryResult.RankedAtRecallK, StringComparer.Ordinal))
            {
                var differenceIndex = FindFirstDifference(originalQueryResult.RankedAtRecallK, restoredQueryResult.RankedAtRecallK);
                return new SnapshotMismatch
                {
                    Message = $"Recall ranking mismatch for configuration '{originalSummary.Name}', query '{queryId}'.",
                    ConfigurationName = originalSummary.Name,
                    QueryId = queryId,
                    RankingKind = "RecallAtK",
                    DifferenceIndex = differenceIndex,
                    OriginalRanking = originalQueryResult.RankedAtRecallK,
                    RestoredRanking = restoredQueryResult.RankedAtRecallK
                };
            }
        }

        if (!AreMetricsEqual(originalSummary.NdcgAt10, restoredSummary.NdcgAt10))
        {
            return new SnapshotMismatch
            {
                Message = $"nDCG@10 mismatch for '{originalSummary.Name}': original={originalSummary.NdcgAt10:F12}, snapshot={restoredSummary.NdcgAt10:F12}.",
                ConfigurationName = originalSummary.Name
            };
        }

        if (!AreMetricsEqual(originalSummary.MapAt10, restoredSummary.MapAt10))
        {
            return new SnapshotMismatch
            {
                Message = $"MAP@10 mismatch for '{originalSummary.Name}': original={originalSummary.MapAt10:F12}, snapshot={restoredSummary.MapAt10:F12}.",
                ConfigurationName = originalSummary.Name
            };
        }

        if (!AreMetricsEqual(originalSummary.RecallAt100, restoredSummary.RecallAt100))
        {
            return new SnapshotMismatch
            {
                Message = $"Recall@100 mismatch for '{originalSummary.Name}': original={originalSummary.RecallAt100:F12}, snapshot={restoredSummary.RecallAt100:F12}.",
                ConfigurationName = originalSummary.Name
            };
        }
    }

    return null;
}

static bool AreMetricsEqual(double original, double restored) =>
    Math.Abs(original - restored) <= 1e-12;

static void PrintSnapshotVerificationReport(
    string displayName,
    int corpusCount,
    int queryCount,
    IReadOnlyList<EvaluationRun> originalRuns,
    IReadOnlyList<EvaluationRun> restoredRuns,
    SnapshotMismatch? mismatch)
{
    ArgumentNullException.ThrowIfNull(displayName);
    ArgumentNullException.ThrowIfNull(originalRuns);
    ArgumentNullException.ThrowIfNull(restoredRuns);

    Console.WriteLine($"{displayName} Snapshot Verification (BEIR)");
    Console.WriteLine(new string('=', displayName.Length + 29));
    Console.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries | {originalRuns.Count} configs");
    Console.WriteLine($"Status: {(mismatch is null ? "PASS" : "FAIL")}");
    Console.WriteLine("Checked: exact ranked IDs at TopK and Recall@100, plus aggregate metrics");
    Console.WriteLine();

    if (mismatch is not null)
    {
        Console.WriteLine(mismatch.Message);

        if (!string.IsNullOrWhiteSpace(mismatch.ConfigurationName))
            Console.WriteLine($"Config: {mismatch.ConfigurationName}");

        if (!string.IsNullOrWhiteSpace(mismatch.QueryId))
            Console.WriteLine($"Query: {mismatch.QueryId}");

        if (!string.IsNullOrWhiteSpace(mismatch.RankingKind))
            Console.WriteLine($"Ranking: {mismatch.RankingKind}");

        if (mismatch.DifferenceIndex is not null)
            Console.WriteLine($"First differing rank: {mismatch.DifferenceIndex.Value + 1}");

        if (mismatch.OriginalRanking is not null && mismatch.RestoredRanking is not null)
        {
            Console.WriteLine(
                $"Original: {FormatRanking(mismatch.OriginalRanking, mismatch.DifferenceIndex)}");
            Console.WriteLine(
                $"Snapshot: {FormatRanking(mismatch.RestoredRanking, mismatch.DifferenceIndex)}");
        }

        return;
    }

    Console.WriteLine($"{"Configuration",-28} {"nDCG@10",8} {"MAP@10",8} {"R@100",10} {"Orig ms",8} {"Snap ms",8}");
    Console.WriteLine(new string('-', 80));

    for (var i = 0; i < originalRuns.Count; i++)
    {
        var original = originalRuns[i].Summary;
        var restored = restoredRuns[i].Summary;

        Console.WriteLine(
            $"{original.Name,-28} {original.NdcgAt10,8:F5} {original.MapAt10,8:F5} {original.RecallAt100,10:F5} {original.AvgQueryTimeMs,8:F1} {restored.AvgQueryTimeMs,8:F1}");
    }
}

static string FormatRanking(IReadOnlyList<string> ranking, int? differenceIndex = null)
{
    ArgumentNullException.ThrowIfNull(ranking);

    if (differenceIndex is null)
        return string.Join(", ", ranking.Take(10));

    var start = Math.Max(0, differenceIndex.Value - 2);
    var count = Math.Min(5, ranking.Count - start);
    return string.Join(", ", ranking.Skip(start).Take(count));
}

static int FindFirstDifference(IReadOnlyList<string> original, IReadOnlyList<string> restored)
{
    ArgumentNullException.ThrowIfNull(original);
    ArgumentNullException.ThrowIfNull(restored);

    var limit = Math.Min(original.Count, restored.Count);
    for (var i = 0; i < limit; i++)
    {
        if (!string.Equals(original[i], restored[i], StringComparison.Ordinal))
            return i;
    }

    return limit;
}

static List<EvaluationConfig> CreateStandardConfigs(bool hasVectorBenchmarks)
{
    var configs = new List<EvaluationConfig>
    {
        new()
        {
            Name = "Lexical-only (BM25)",
            UseLexical = true,
            UseVector = false,
            TopK = 10,
            LexicalK = 100,
            VectorK = 100,
            LexicalWeight = 1f,
            VectorWeight = 0f,
            RecallAtK = 100
        }
    };

    if (hasVectorBenchmarks)
    {
        configs.Add(new EvaluationConfig
        {
            Name = "Vector-only",
            UseLexical = false,
            UseVector = true,
            TopK = 10,
            LexicalK = 100,
            VectorK = 100,
            LexicalWeight = 0f,
            VectorWeight = 1f,
            RecallAtK = 100
        });

        configs.Add(new EvaluationConfig
        {
            Name = "Hybrid (BM25 + Vector)",
            UseLexical = true,
            UseVector = true,
            TopK = 10,
            LexicalK = 100,
            VectorK = 100,
            LexicalWeight = 0.5f,
            VectorWeight = 1f,
            RecallAtK = 100
        });
    }

    return configs;
}

static List<EvaluationConfig> CreateSweepConfigs(bool hasVectorBenchmarks)
{
    // Sweep grid: ~200 configs for hybrid (~10 min), ~12 for lexical-only
    float[] lexicalWeights = hasVectorBenchmarks
        ? [0f, 0.1f, 0.3f, 0.5f, 1f, 1.5f]
        : [1f];
    float[] vectorWeights = hasVectorBenchmarks
        ? [0f, 0.1f, 0.3f, 0.5f, 1f, 1.5f]
        : [0f];
    int[] rrfKValues = hasVectorBenchmarks
        ? [1, 20, 60]
        : [60];
    float[] titleBoosts = [0.5f, 1f, 2f];

    const int candidateK = 100;

    var configs = new List<EvaluationConfig>();

    foreach (var lexicalWeight in lexicalWeights)
    foreach (var vectorWeight in vectorWeights)
    foreach (var rrfK in rrfKValues)
    foreach (var titleBoost in titleBoosts)
    {
        if (lexicalWeight == 0f && vectorWeight == 0f)
            continue;

        if (vectorWeight > 0f && !hasVectorBenchmarks)
            continue;

        var useLexical = lexicalWeight > 0f;
        var useVector = vectorWeight > 0f;

        if ((!useLexical || !useVector) && rrfK != 60)
            continue;

        if (!useLexical && titleBoost != 1f)
            continue;

        configs.Add(new EvaluationConfig
        {
            Name = $"L={lexicalWeight:F1} V={vectorWeight:F1} k={rrfK} tb={titleBoost:F1}",
            UseLexical = useLexical,
            UseVector = useVector,
            TopK = 10,
            LexicalK = candidateK,
            VectorK = candidateK,
            LexicalWeight = lexicalWeight,
            VectorWeight = vectorWeight,
            RrfK = rrfK,
            RecallAtK = 100,
            TitleBoost = titleBoost
        });
    }

    return configs;
}

static IReadOnlyList<Document> BuildDocuments(
    IReadOnlyDictionary<string, (string Title, string Text)> corpus,
    IReadOnlyDictionary<string, float[]>? embeddings)
{
    ArgumentNullException.ThrowIfNull(corpus);

    var docs = new List<Document>(corpus.Count);
    foreach (var (id, payload) in corpus)
    {
        float[]? vector = null;
        if (embeddings is not null)
            embeddings.TryGetValue(id, out vector);

        docs.Add(new Document
        {
            Id = id,
            Title = payload.Title,
            Body = payload.Text,
            Embedding = vector
        });
    }

    return docs;
}

static bool HasQueryEmbeddings(
    IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> qrels,
    IReadOnlyDictionary<string, float[]> embeddings)
{
    ArgumentNullException.ThrowIfNull(qrels);
    ArgumentNullException.ThrowIfNull(embeddings);

    foreach (var queryId in qrels.Keys)
    {
        if (!embeddings.ContainsKey(queryId))
            return false;
    }

    return true;
}

static ParsedOptions ParseArguments(string[] args)
{
    ArgumentNullException.ThrowIfNull(args);

    var dataset = "nfcorpus";
    var dataDir = "./benchmarks/data";
    string? embeddingsPath = null;
    var sweep = false;
    var verifySnapshotRoundTrip = false;

    for (var i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--dataset":
            case "-d":
                if (i + 1 >= args.Length)
                    throw new ArgumentException("Missing value for --dataset.");
                dataset = args[++i];
                break;

            case "--data-dir":
                if (i + 1 >= args.Length)
                    throw new ArgumentException("Missing value for --data-dir.");
                dataDir = args[++i];
                break;

            case "--embeddings":
                if (i + 1 >= args.Length)
                    throw new ArgumentException("Missing value for --embeddings.");
                embeddingsPath = args[++i];
                break;

            case "--sweep":
                sweep = true;
                break;

            case "--verify-snapshot-roundtrip":
                verifySnapshotRoundTrip = true;
                break;

            case "--list-datasets":
                PrintAvailableDatasets();
                Environment.Exit(0);
                break;

            case "--help":
            case "-h":
                PrintUsage();
                Environment.Exit(0);
                break;

            default:
                throw new ArgumentException($"Unknown argument: {args[i]}");
        }
    }

    return new ParsedOptions
    {
        Dataset = dataset,
        DataDir = dataDir,
        EmbeddingsPath = embeddingsPath,
        Sweep = sweep,
        VerifySnapshotRoundTrip = verifySnapshotRoundTrip
    };
}

static void PrintUsage()
{
    Console.WriteLine("Retrievo.Benchmarks — BEIR dataset evaluation");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run --project benchmarks/Retrievo.Benchmarks -- [options]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --dataset, -d <name>              BEIR dataset to evaluate (default: nfcorpus)");
    Console.WriteLine("  --data-dir <dir>                  Base directory for dataset storage (default: ./benchmarks/data)");
    Console.WriteLine("  --embeddings <path>               Path to pre-computed embeddings binary cache");
    Console.WriteLine("  --sweep                           Run parameter sweep (grid search over fusion params)");
    Console.WriteLine("  --verify-snapshot-roundtrip       Export a snapshot, re-import it, and assert identical benchmark outputs");
    Console.WriteLine("  --list-datasets                   Show available BEIR datasets and exit");
    Console.WriteLine("  --help, -h                        Show this help and exit");
    Console.WriteLine();
    Console.WriteLine("Examples:");
    Console.WriteLine("  dotnet run --project benchmarks/Retrievo.Benchmarks");
    Console.WriteLine("  dotnet run --project benchmarks/Retrievo.Benchmarks -- --dataset scifact");
    Console.WriteLine("  dotnet run --project benchmarks/Retrievo.Benchmarks -- --dataset nfcorpus --embeddings embeddings.bin --sweep");
    Console.WriteLine("  dotnet run --project benchmarks/Retrievo.Benchmarks -- --dataset nfcorpus --embeddings benchmarks/fixtures/embeddings/nfcorpus.text-embedding-3-small.cache --verify-snapshot-roundtrip");
}

static void PrintAvailableDatasets()
{
    Console.WriteLine("Available BEIR datasets:");
    Console.WriteLine();
    foreach (var (id, info) in BeirDatasetDownloader.KnownDatasets.OrderBy(kv => kv.Key, StringComparer.Ordinal))
    {
        Console.WriteLine($"  {id,-16} {info.Description}");
    }
    Console.WriteLine();
    Console.WriteLine("Any BEIR dataset ID can be used, even if not listed above.");
    Console.WriteLine("The tool will attempt to download it from the BEIR CDN.");
}

static void PrintResults(string datasetId, int corpusCount, int queryCount, IReadOnlyList<EvaluationResult> results)
{
    var displayName = BeirDatasetDownloader.KnownDatasets.TryGetValue(datasetId, out var info)
        ? info.DisplayName
        : datasetId;

    Console.WriteLine($"{displayName} Benchmark (BEIR)");
    Console.WriteLine(new string('=', displayName.Length + 17));
    Console.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries");
    Console.WriteLine();
    Console.WriteLine("Configuration            nDCG@10   MAP@10   Recall@100  Avg Query");
    Console.WriteLine("-------------------------------------------------------------------");

    foreach (var result in results)
    {
        Console.WriteLine(
            $"{result.Name,-24} {result.NdcgAt10,8:F5} {result.MapAt10,8:F5} {result.RecallAt100,10:F5}  {result.AvgQueryTimeMs,8:F1}ms");
    }
}

file sealed record ParsedOptions
{
    public required string Dataset { get; init; }

    public required string DataDir { get; init; }

    public string? EmbeddingsPath { get; init; }

    public bool Sweep { get; init; }

    public bool VerifySnapshotRoundTrip { get; init; }
}

file sealed record SnapshotMismatch
{
    public required string Message { get; init; }

    public string? ConfigurationName { get; init; }

    public string? QueryId { get; init; }

    public string? RankingKind { get; init; }

    public int? DifferenceIndex { get; init; }

    public IReadOnlyList<string>? OriginalRanking { get; init; }

    public IReadOnlyList<string>? RestoredRanking { get; init; }
}
