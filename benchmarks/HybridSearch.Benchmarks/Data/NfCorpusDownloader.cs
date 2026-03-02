using System.IO.Compression;

namespace HybridSearch.Benchmarks.Data;

/// <summary>
/// Downloads and extracts the NFCorpus dataset from the BEIR repository.
/// </summary>
public static class NfCorpusDownloader
{
    private const string DownloadUrl = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip";

    /// <summary>
    /// Ensures NFCorpus data is available at the specified directory.
    /// Downloads and extracts if not already present.
    /// </summary>
    /// <param name="dataDir">Base directory where nfcorpus/ should exist.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Path to extracted nfcorpus/ directory.</returns>
    public static async Task<string> EnsureDataAsync(string dataDir, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(dataDir);

        var fullDataDir = Path.GetFullPath(dataDir);
        Directory.CreateDirectory(fullDataDir);

        var datasetDir = Path.Combine(fullDataDir, "nfcorpus");
        var corpusPath = Path.Combine(datasetDir, "corpus.jsonl");
        if (File.Exists(corpusPath))
        {
            Console.Error.WriteLine($"NFCorpus already present: {datasetDir}");
            return datasetDir;
        }

        var zipPath = Path.Combine(fullDataDir, "nfcorpus.zip");

        Console.Error.WriteLine($"Downloading NFCorpus from {DownloadUrl}");

        using var httpClient = new HttpClient();
        using var response = await httpClient.GetAsync(DownloadUrl, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength;
        await using (var source = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
        await using (var destination = new FileStream(zipPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true))
        {
            var buffer = new byte[81920];
            long downloaded = 0;
            DateTime nextProgressUpdateUtc = DateTime.UtcNow;

            while (true)
            {
                var read = await source.ReadAsync(buffer.AsMemory(0, buffer.Length), ct).ConfigureAwait(false);
                if (read == 0)
                    break;

                await destination.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                downloaded += read;

                if (DateTime.UtcNow >= nextProgressUpdateUtc)
                {
                    if (totalBytes is > 0)
                    {
                        var percent = downloaded * 100d / totalBytes.Value;
                        Console.Error.WriteLine($"Download progress: {percent:F1}% ({downloaded:N0}/{totalBytes.Value:N0} bytes)");
                    }
                    else
                    {
                        Console.Error.WriteLine($"Download progress: {downloaded:N0} bytes");
                    }

                    nextProgressUpdateUtc = DateTime.UtcNow.AddMilliseconds(500);
                }
            }
        }

        Console.Error.WriteLine("Download completed. Extracting archive...");

        if (Directory.Exists(datasetDir))
            Directory.Delete(datasetDir, recursive: true);

        ZipFile.ExtractToDirectory(zipPath, fullDataDir, overwriteFiles: true);
        File.Delete(zipPath);

        if (!File.Exists(corpusPath))
            throw new DirectoryNotFoundException($"NFCorpus extraction failed. Missing file: {corpusPath}");

        Console.Error.WriteLine($"NFCorpus ready: {datasetDir}");
        return datasetDir;
    }
}
