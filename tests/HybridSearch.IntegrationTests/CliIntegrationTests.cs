using System.Diagnostics;

namespace HybridSearch.IntegrationTests;

/// <summary>
/// Integration tests for the CLI tool.
/// These tests invoke the CLI as a subprocess and verify output and exit codes.
/// </summary>
public class CliIntegrationTests : IDisposable
{
    private readonly string _testDocsFolder;
    private readonly string _cliProjectPath;

    public CliIntegrationTests()
    {
        // Create temp folder with test documents
        _testDocsFolder = Path.Combine(Path.GetTempPath(), $"hybridsearch-cli-test-{Guid.NewGuid():N}");
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

        // Resolve CLI project path relative to test assembly
        var assemblyDir = Path.GetDirectoryName(typeof(CliIntegrationTests).Assembly.Location)!;
        // Navigate from tests/HybridSearch.IntegrationTests/bin/Debug/net8.0 up to repo root
        _cliProjectPath = Path.GetFullPath(Path.Combine(assemblyDir, "..", "..", "..", "..", "..", "src", "HybridSearch.Cli", "HybridSearch.Cli.csproj"));
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
        Assert.Contains("Folder not found", stderr);
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

    private async Task<(int ExitCode, string Stdout, string Stderr)> RunCliAsync(string arguments)
    {
        var psi = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = $"run --project \"{_cliProjectPath}\" --no-build -- {arguments}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        // Clear Azure OpenAI env vars to ensure lexical-only mode
        psi.Environment["HYBRIDSEARCH_AZURE_OPENAI_ENDPOINT"] = "";
        psi.Environment["HYBRIDSEARCH_AZURE_OPENAI_KEY"] = "";
        psi.Environment["HYBRIDSEARCH_AZURE_OPENAI_DEPLOYMENT"] = "";

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
