using System.CommandLine;
using System.CommandLine.Invocation;
using System.Text.Json;
using Retrievo;
using Retrievo.Abstractions;
using Retrievo.AzureOpenAI;
using Retrievo.Models;

var inputArgument = new Argument<string>(
    name: "input",
    description: "Path to a folder of .md/.txt files, or a saved Retrievo snapshot file.");

var folderArgument = new Argument<DirectoryInfo>(
    name: "folder",
    description: "Path to the folder containing .md and .txt files to snapshot.");

var textOption = new Option<string>(
    name: "--text",
    description: "The text query to search for.")
{ IsRequired = true };
textOption.AddAlias("-t");

var topKOption = new Option<int>(
    name: "--top-k",
    getDefaultValue: () => 10,
    description: "Number of results to return.");
topKOption.AddAlias("-k");

var explainOption = new Option<bool>(
    name: "--explain",
    description: "Show detailed score breakdown for each result.");

var embeddingProviderOption = new Option<string?>(
    name: "--embedding-provider",
    description: "Embedding provider to use. Supported: 'azure-openai'. " +
                 "If not specified, text queries operate in lexical-only mode even when a snapshot contains stored embeddings.");

var outputOption = new Option<FileInfo>(
    name: "--output",
    description: "Destination snapshot file path.")
{ IsRequired = true };
outputOption.AddAlias("-o");

var queryCommand = new Command("query", "Query a folder of local documents or a previously exported snapshot.")
{
    inputArgument,
    textOption,
    topKOption,
    explainOption,
    embeddingProviderOption
};

queryCommand.SetHandler(async (InvocationContext context) =>
{
    var inputPath = context.ParseResult.GetValueForArgument(inputArgument)!;
    var text = context.ParseResult.GetValueForOption(textOption)!;
    var topK = context.ParseResult.GetValueForOption(topKOption);
    var explain = context.ParseResult.GetValueForOption(explainOption);
    var providerName = context.ParseResult.GetValueForOption(embeddingProviderOption);
    var input = ResolveInput(inputPath);

    if (!input.Exists)
    {
        Console.Error.WriteLine($"Error: Input not found: {input.FullName}");
        context.ExitCode = 1;
        return;
    }

    // Configure embedding provider
    IEmbeddingProvider? embeddingProvider = null;
    if (!string.IsNullOrEmpty(providerName))
    {
        embeddingProvider = CreateEmbeddingProvider(providerName);
        if (embeddingProvider is null)
        {
            context.ExitCode = 1;
            return;
        }
    }
    else
    {
        // Try auto-detect from environment variables
        embeddingProvider = TryAutoDetectProvider();
    }

    if (input is DirectoryInfo)
    {
        Console.Error.WriteLine($"Indexing files from: {input.FullName}");
    }
    else
    {
        Console.Error.WriteLine($"Loading snapshot from: {input.FullName}");
    }

    HybridSearchIndex index;
    try
    {
        index = await LoadIndexAsync(input, embeddingProvider, context.GetCancellationToken());
    }
    catch (InvalidOperationException) when (input is DirectoryInfo)
    {
        Console.Error.WriteLine($"Error: No indexable documents found in {input.FullName}");
        context.ExitCode = 1;
        return;
    }
    catch (InvalidOperationException ex)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
        context.ExitCode = 1;
        return;
    }
    catch (DirectoryNotFoundException)
    {
        Console.Error.WriteLine($"Error: Folder not found: {input.FullName}");
        context.ExitCode = 1;
        return;
    }
    catch (InvalidDataException ex)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
        context.ExitCode = 1;
        return;
    }
    catch (JsonException ex)
    {
        Console.Error.WriteLine($"Error: Invalid snapshot JSON. {ex.Message}");
        context.ExitCode = 1;
        return;
    }
    catch (NotSupportedException ex)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
        context.ExitCode = 1;
        return;
    }

    using (index)
    {
        var stats = index.GetStats();
        if (embeddingProvider is null)
        {
            Console.Error.WriteLine("Warning: No embedding provider configured. Text queries will run in lexical-only mode.");
            Console.Error.WriteLine("  Stored document embeddings can still be loaded from snapshots, but query vectors");
            Console.Error.WriteLine("  will not be generated unless you configure an embedding provider.");
            Console.Error.WriteLine("  Set RETRIEVO_AZURE_OPENAI_ENDPOINT, RETRIEVO_AZURE_OPENAI_KEY, and");
            Console.Error.WriteLine("  RETRIEVO_AZURE_OPENAI_DEPLOYMENT environment variables, or use");
            Console.Error.WriteLine("  --embedding-provider azure-openai.");
            Console.Error.WriteLine();
        }

        var action = input is DirectoryInfo ? "Indexed" : "Loaded";
        Console.Error.WriteLine($"{action} {stats.DocumentCount} documents " +
            (stats.EmbeddingDimension.HasValue
                ? $"({stats.EmbeddingDimension.Value}-dim embeddings) "
                : "(lexical only) ") +
            $"in {stats.IndexBuildTimeMs:F1}ms");
        Console.Error.WriteLine();

        // Execute query
        var query = new HybridQuery
        {
            Text = text,
            TopK = topK,
            Explain = explain
        };

        var response = await index.SearchAsync(query, context.GetCancellationToken());

        // Output results
        if (response.Results.Count == 0)
        {
            Console.WriteLine("No results found.");
        }
        else
        {
            Console.WriteLine($"Found {response.Results.Count} results in {response.QueryTimeMs:F1}ms:");
            Console.WriteLine();

            for (int i = 0; i < response.Results.Count; i++)
            {
                var result = response.Results[i];
                Console.WriteLine($"  [{i + 1}] {result.Id}  (score: {result.Score:F6})");

                if (explain && result.Explain is not null)
                {
                    var ex = result.Explain;
                    Console.Write("      ");
                    if (ex.LexicalRank.HasValue)
                        Console.Write($"lexical: rank={ex.LexicalRank.Value} contrib={ex.LexicalContribution:F6}  ");
                    else
                        Console.Write("lexical: n/a  ");

                    if (ex.VectorRank.HasValue)
                        Console.Write($"vector: rank={ex.VectorRank.Value} contrib={ex.VectorContribution:F6}");
                    else
                        Console.Write("vector: n/a");

                    Console.WriteLine($"  fused={ex.FusedScore:F6}");
                }
            }
        }
    }
});

var exportCommand = new Command("export", "Build an index from a folder and export it as a snapshot.")
{
    folderArgument,
    outputOption,
    embeddingProviderOption
};

exportCommand.SetHandler(async (InvocationContext context) =>
{
    var folder = context.ParseResult.GetValueForArgument(folderArgument);
    var output = context.ParseResult.GetValueForOption(outputOption)!;
    var providerName = context.ParseResult.GetValueForOption(embeddingProviderOption);

    if (!folder.Exists)
    {
        Console.Error.WriteLine($"Error: Folder not found: {folder.FullName}");
        context.ExitCode = 1;
        return;
    }

    IEmbeddingProvider? embeddingProvider = null;
    if (!string.IsNullOrEmpty(providerName))
    {
        embeddingProvider = CreateEmbeddingProvider(providerName);
        if (embeddingProvider is null)
        {
            context.ExitCode = 1;
            return;
        }
    }
    else
    {
        embeddingProvider = TryAutoDetectProvider();
    }

    Console.Error.WriteLine($"Indexing files from: {folder.FullName}");
    var builder = new HybridSearchIndexBuilder()
        .AddFolder(folder.FullName);

    if (embeddingProvider is not null)
    {
        builder.WithEmbeddingProvider(embeddingProvider);
    }

    HybridSearchIndex index;
    try
    {
        index = await builder.BuildAsync(context.GetCancellationToken());
    }
    catch (InvalidOperationException)
    {
        Console.Error.WriteLine($"Error: No indexable documents found in {folder.FullName}");
        context.ExitCode = 1;
        return;
    }

    using (index)
    {
        try
        {
            await index.ExportSnapshotAsync(output.FullName, context.GetCancellationToken());
        }
        catch (DirectoryNotFoundException)
        {
            Console.Error.WriteLine($"Error: Output directory not found for {output.FullName}");
            context.ExitCode = 1;
            return;
        }

        var stats = index.GetStats();
        Console.WriteLine($"Exported snapshot with {stats.DocumentCount} documents to {output.FullName}");
    }
});

var rootCommand = new RootCommand("HybridSearch CLI — hybrid lexical + vector search for local folders and saved snapshots.")
{
    queryCommand,
    exportCommand
};

return await rootCommand.InvokeAsync(args);

// --- Helper methods ---

static IEmbeddingProvider? CreateEmbeddingProvider(string providerName)
{
    if (string.Equals(providerName, "azure-openai", StringComparison.OrdinalIgnoreCase))
    {
        var endpoint = Environment.GetEnvironmentVariable("RETRIEVO_AZURE_OPENAI_ENDPOINT");
        var key = Environment.GetEnvironmentVariable("RETRIEVO_AZURE_OPENAI_KEY");
        var deployment = Environment.GetEnvironmentVariable("RETRIEVO_AZURE_OPENAI_DEPLOYMENT");

        if (string.IsNullOrEmpty(endpoint) || string.IsNullOrEmpty(key) || string.IsNullOrEmpty(deployment))
        {
            Console.Error.WriteLine("Error: Azure OpenAI embedding provider requires these environment variables:");
            Console.Error.WriteLine("  RETRIEVO_AZURE_OPENAI_ENDPOINT");
            Console.Error.WriteLine("  RETRIEVO_AZURE_OPENAI_KEY");
            Console.Error.WriteLine("  RETRIEVO_AZURE_OPENAI_DEPLOYMENT");
            return null;
        }

        return new AzureOpenAIEmbeddingProvider(new Uri(endpoint), key, deployment);
    }

    Console.Error.WriteLine($"Error: Unknown embedding provider '{providerName}'. Supported: azure-openai");
    return null;
}

static IEmbeddingProvider? TryAutoDetectProvider()
{
    var endpoint = Environment.GetEnvironmentVariable("RETRIEVO_AZURE_OPENAI_ENDPOINT");
    var key = Environment.GetEnvironmentVariable("RETRIEVO_AZURE_OPENAI_KEY");
    var deployment = Environment.GetEnvironmentVariable("RETRIEVO_AZURE_OPENAI_DEPLOYMENT");

    if (!string.IsNullOrEmpty(endpoint) && !string.IsNullOrEmpty(key) && !string.IsNullOrEmpty(deployment))
    {
        Console.Error.WriteLine("Auto-detected Azure OpenAI embedding provider from environment variables.");
        return new AzureOpenAIEmbeddingProvider(new Uri(endpoint), key, deployment);
    }

    return null;
}

static async Task<HybridSearchIndex> LoadIndexAsync(
    FileSystemInfo input,
    IEmbeddingProvider? embeddingProvider,
    CancellationToken ct)
{
    if (input is DirectoryInfo directory)
    {
        var builder = new HybridSearchIndexBuilder()
            .AddFolder(directory.FullName);

        if (embeddingProvider is not null)
        {
            builder.WithEmbeddingProvider(embeddingProvider);
        }

        return await builder.BuildAsync(ct);
    }

    if (input is FileInfo file)
    {
        return await HybridSearchIndex.ImportSnapshotAsync(file.FullName, embeddingProvider, ct: ct);
    }

    throw new InvalidOperationException($"Unsupported input type: {input.GetType().Name}");
}

static FileSystemInfo ResolveInput(string path)
{
    ArgumentException.ThrowIfNullOrWhiteSpace(path);

    if (Directory.Exists(path))
        return new DirectoryInfo(path);

    return new FileInfo(path);
}
