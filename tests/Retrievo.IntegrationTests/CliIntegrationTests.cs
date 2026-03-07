using System.Diagnostics;
using Retrievo.Models;

namespace Retrievo.IntegrationTests;

/// <summary>
/// Integration tests for the CLI tool.
/// These tests invoke the CLI as a subprocess and verify output and exit codes.
/// </summary>
public class CliIntegrationTests : IDisposable
{
    private readonly string _testDocsFolder;
    private readonly string _cliDllPath;

    public CliIntegrationTests()
    {
        // Create temp folder with test documents
        _testDocsFolder = Path.Combine(Path.GetTempPath(), $"retrievo-cli-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDocsFolder);

        File.WriteAllText(Path.Combine(_testDocsFolder, "machine-learning.md"),
            "# Machine Learning\nNeural networks learn complex patterns from training data. " +
            "Deep learning has transformed the field of artificial intelligence.");

        File.WriteAllText(Path.Combine(_testDocsFolder, "databases.txt"),
            "SQL queries retrieve structured data from relational tables. " +
            "Indexing accelerates query performance on large datasets.");

        File.WriteAllText(Path.Combine(_testDocsFolder, "security.md"),
            "# Security Best Practices\nEncryption protects sensitive data at rest and in transit. " +
            "Authentication verifies user identity with credentials.");

        // Derive the active configuration and TFM from the current test assembly output path
        // so these tests work under both Debug/Release and future target framework changes.
        var assemblyDir = Path.GetDirectoryName(typeof(CliIntegrationTests).Assembly.Location)!;
        var targetFramework = new DirectoryInfo(assemblyDir).Name;
        var configurationDir = Directory.GetParent(assemblyDir)
            ?? throw new InvalidOperationException($"Could not determine build configuration from '{assemblyDir}'.");
        var configuration = configurationDir.Name;
        var repoRoot = Path.GetFullPath(Path.Combine(assemblyDir, "..", "..", "..", "..", ".."));

        _cliDllPath = Path.Combine(repoRoot, "src", "Retrievo.Cli", "bin", configuration, targetFramework, "Retrievo.Cli.dll");

        if (!File.Exists(_cliDllPath))
        {
            throw new FileNotFoundException(
                $"CLI assembly not found at '{_cliDllPath}'. Ensure the integration test project builds the CLI for configuration '{configuration}' and target framework '{targetFramework}' before running tests.",
                _cliDllPath);
        }
    }

    /// <summary>
    /// Spec §5.2 CLI tests: "CLI query smoke: hybridsearch query <folder> --text '...'
    /// returns non-empty output and exit code 0."
    /// </summary>
    [Fact]
    public async Task CliQuerySmoke_ReturnsNonEmptyOutput_ExitCode0()
    {
        var (exitCode, stdout, stderr) = await RunCliAsync(
            $"query \"{_testDocsFolder}\" --text \"neural network\"");

        Assert.Equal(0, exitCode);
        Assert.Contains("Found", stdout);
        Assert.Contains("machine-learning.md", stdout);
    }

    /// <summary>
    /// Spec §5.2 CLI tests: "--explain prints lexical/vector ranks for returned docs."
    /// </summary>
    [Fact]
    public async Task CliExplain_PrintsRanksForReturnedDocs()
    {
        var (exitCode, stdout, stderr) = await RunCliAsync(
            $"query \"{_testDocsFolder}\" --text \"neural network\" --explain");

        Assert.Equal(0, exitCode);
        Assert.Contains("Found", stdout);
        Assert.Contains("lexical:", stdout);
        Assert.Contains("rank=", stdout);
        Assert.Contains("contrib=", stdout);
        Assert.Contains("fused=", stdout);
    }

    /// <summary>
    /// CLI with --top-k flag limits results.
    /// </summary>
    [Fact]
    public async Task CliTopK_LimitsResults()
    {
        var (exitCode, stdout, _) = await RunCliAsync(
            $"query \"{_testDocsFolder}\" --text \"data\" -k 1");

        Assert.Equal(0, exitCode);
        Assert.Contains("Found 1 result", stdout);
    }

    /// <summary>
    /// CLI with nonexistent folder returns exit code 1.
    /// </summary>
    [Fact]
    public async Task CliNonExistentFolder_ExitCode1()
    {
        var nonExistent = Path.Combine(Path.GetTempPath(), $"does-not-exist-{Guid.NewGuid():N}");
        var (exitCode, _, stderr) = await RunCliAsync(
            $"query \"{nonExistent}\" --text \"test\"");

        Assert.Equal(1, exitCode);
        Assert.Contains("Input not found", stderr);
    }

    /// <summary>
    /// CLI without embedding provider shows lexical-only warning on stderr.
    /// </summary>
    [Fact]
    public async Task CliNoEmbeddingProvider_ShowsLexicalOnlyWarning()
    {
        var (exitCode, _, stderr) = await RunCliAsync(
            $"query \"{_testDocsFolder}\" --text \"encryption\"");

        Assert.Equal(0, exitCode);
        Assert.Contains("lexical-only mode", stderr);
    }

    /// <summary>
    /// CLI can export a snapshot and query it later.
    /// </summary>
    [Fact]
    public async Task CliExport_ThenQuerySnapshot_Works()
    {
        var snapshotPath = Path.Combine(_testDocsFolder, "snapshot.retrievo.json");

        var (exportExitCode, exportStdout, exportStderr) = await RunCliAsync(
            $"export \"{_testDocsFolder}\" --output \"{snapshotPath}\"");

        Assert.Equal(0, exportExitCode);
        Assert.Contains("Exported snapshot", exportStdout);
        Assert.True(File.Exists(snapshotPath));
        Assert.Contains("Indexing files from", exportStderr);

        var (queryExitCode, queryStdout, queryStderr) = await RunCliAsync(
            $"query \"{snapshotPath}\" --text \"neural network\"");

        Assert.Equal(0, queryExitCode);
        Assert.Contains("Found", queryStdout);
        Assert.Contains("machine-learning.md", queryStdout);
        Assert.Contains("Loading snapshot from", queryStderr);
    }

    /// <summary>
    /// Snapshot queries without an embedding provider warn that text queries run in lexical-only mode,
    /// even when the snapshot contains stored document embeddings.
    /// </summary>
    [Fact]
    public async Task CliQuery_SnapshotWithoutEmbeddingProvider_ShowsLexicalOnlyWarning()
    {
        var snapshotPath = Path.Combine(_testDocsFolder, "snapshot-with-embeddings.retrievo.json");

        using (var index = new HybridSearchIndexBuilder()
            .AddDocument(new Document
            {
                Id = "doc-1",
                Title = "Vector Search",
                Body = "Neural retrieval uses embeddings to compare semantic similarity.",
                Embedding = new float[] { 1f, 0f }
            })
            .AddDocument(new Document
            {
                Id = "doc-2",
                Title = "Lexical Search",
                Body = "Keyword search relies on BM25 lexical ranking.",
                Embedding = new float[] { 0f, 1f }
            })
            .Build())
        {
            index.ExportSnapshot(snapshotPath);
        }

        var (exitCode, _, stderr) = await RunCliAsync(
            $"query \"{snapshotPath}\" --text \"semantic similarity\"");

        Assert.Equal(0, exitCode);
        Assert.Contains("Loading snapshot from", stderr);
        Assert.Contains("Text queries will run in lexical-only mode", stderr);
        Assert.Contains("Stored document embeddings can still be loaded from snapshots", stderr);
    }

    /// <summary>
    /// CLI query reports invalid snapshot files with a non-zero exit code.
    /// </summary>
    [Fact]
    public async Task CliQuery_InvalidSnapshot_ExitCode1()
    {
        var snapshotPath = Path.Combine(_testDocsFolder, "invalid.retrievo.json");
        await File.WriteAllTextAsync(snapshotPath, "{ not valid json }");

        var (exitCode, _, stderr) = await RunCliAsync(
            $"query \"{snapshotPath}\" --text \"neural network\"");

        Assert.Equal(1, exitCode);
        Assert.Contains("Invalid snapshot JSON", stderr);
    }

    private async Task<(int ExitCode, string Stdout, string Stderr)> RunCliAsync(string arguments)
    {
        var psi = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = $"\"{_cliDllPath}\" {arguments}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        // Clear Azure OpenAI env vars to ensure lexical-only mode
        psi.Environment["RETRIEVO_AZURE_OPENAI_ENDPOINT"] = "";
        psi.Environment["RETRIEVO_AZURE_OPENAI_KEY"] = "";
        psi.Environment["RETRIEVO_AZURE_OPENAI_DEPLOYMENT"] = "";

        using var process = Process.Start(psi)!;
        var stdout = await process.StandardOutput.ReadToEndAsync();
        var stderr = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        return (process.ExitCode, stdout, stderr);
    }

    public void Dispose()
    {
        try { Directory.Delete(_testDocsFolder, recursive: true); }
        catch { /* best effort */ }
    }
}
