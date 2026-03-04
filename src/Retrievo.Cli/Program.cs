using System.CommandLine;
using System.CommandLine.Invocation;
using Retrievo;
using Retrievo.Abstractions;
using Retrievo.AzureOpenAI;
using Retrievo.Models;

var folderArgument = new Argument<DirectoryInfo>(
    name: "folder",
    description: "Path to the folder containing .md and .txt files to search.");

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
                 "If not specified, operates in lexical-only mode.");

var queryCommand = new Command("query", "Build an in-memory index from a folder and run a hybrid search query.")
{
    folderArgument,
    textOption,
    topKOption,
    explainOption,
    embeddingProviderOption
};

queryCommand.SetHandler(async (InvocationContext context) =>
{
    var folder = context.ParseResult.GetValueForArgument(folderArgument);
    var text = context.ParseResult.GetValueForOption(textOption)!;
    var topK = context.ParseResult.GetValueForOption(topKOption);
    var explain = context.ParseResult.GetValueForOption(explainOption);
    var providerName = context.ParseResult.GetValueForOption(embeddingProviderOption);

    if (!folder.Exists)
    {
        Console.Error.WriteLine($"Error: Folder not found: {folder.FullName}");
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

    if (embeddingProvider is null)
    {
        Console.Error.WriteLine("Warning: No embedding provider configured. Operating in lexical-only mode.");
        Console.Error.WriteLine("  Set RETRIEVO_AZURE_OPENAI_ENDPOINT, RETRIEVO_AZURE_OPENAI_KEY, and");
        Console.Error.WriteLine("  RETRIEVO_AZURE_OPENAI_DEPLOYMENT environment variables, or use");
        Console.Error.WriteLine("  --embedding-provider azure-openai.");
        Console.Error.WriteLine();
    }

    // Build the index
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
        var stats = index.GetStats();
        Console.Error.WriteLine($"Indexed {stats.DocumentCount} documents " +
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

var rootCommand = new RootCommand("HybridSearch CLI — hybrid lexical + vector search for local document folders.")
{
    queryCommand
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
