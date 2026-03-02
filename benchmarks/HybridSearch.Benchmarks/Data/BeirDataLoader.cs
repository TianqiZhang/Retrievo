using System.Text.Json;

namespace HybridSearch.Benchmarks.Data;

/// <summary>
/// Loads BEIR-format datasets (corpus.jsonl, queries.jsonl, qrels/*.tsv).
/// </summary>
public static class BeirDataLoader
{
    /// <summary>
    /// Load corpus.jsonl from a BEIR dataset.
    /// </summary>
    /// <param name="path">Path to corpus.jsonl.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A dictionary of document ID to (title, text).</returns>
    public static async Task<IReadOnlyDictionary<string, (string Title, string Text)>> LoadCorpusAsync(string path, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(path);

        if (!File.Exists(path))
            throw new FileNotFoundException($"Corpus file not found: {path}", path);

        var corpus = new Dictionary<string, (string Title, string Text)>(StringComparer.Ordinal);

        await using var stream = File.OpenRead(path);
        using var reader = new StreamReader(stream);

        while (true)
        {
            ct.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
            if (line is null)
                break;
            if (string.IsNullOrWhiteSpace(line))
                continue;

            using var json = JsonDocument.Parse(line);
            var root = json.RootElement;

            var id = root.GetProperty("_id").GetString();
            var title = root.GetProperty("title").GetString();
            var text = root.GetProperty("text").GetString();

            if (string.IsNullOrWhiteSpace(id))
                throw new InvalidDataException("Invalid corpus.jsonl line: _id is missing or empty.");
            if (text is null)
                throw new InvalidDataException($"Invalid corpus.jsonl line for '{id}': text is missing.");

            corpus[id] = (title ?? string.Empty, text);
        }

        return corpus;
    }

    /// <summary>
    /// Load queries.jsonl from a BEIR dataset.
    /// </summary>
    /// <param name="path">Path to queries.jsonl.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A dictionary of query ID to query text.</returns>
    public static async Task<IReadOnlyDictionary<string, string>> LoadQueriesAsync(string path, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(path);

        if (!File.Exists(path))
            throw new FileNotFoundException($"Queries file not found: {path}", path);

        var queries = new Dictionary<string, string>(StringComparer.Ordinal);

        await using var stream = File.OpenRead(path);
        using var reader = new StreamReader(stream);

        while (true)
        {
            ct.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
            if (line is null)
                break;
            if (string.IsNullOrWhiteSpace(line))
                continue;

            using var json = JsonDocument.Parse(line);
            var root = json.RootElement;

            var id = root.GetProperty("_id").GetString();
            var text = root.GetProperty("text").GetString();

            if (string.IsNullOrWhiteSpace(id))
                throw new InvalidDataException("Invalid queries.jsonl line: _id is missing or empty.");
            if (string.IsNullOrWhiteSpace(text))
                throw new InvalidDataException($"Invalid queries.jsonl line for '{id}': text is missing or empty.");

            queries[id] = text;
        }

        return queries;
    }

    /// <summary>
    /// Load qrels/test.tsv from a BEIR dataset.
    /// </summary>
    /// <param name="path">Path to qrels TSV file.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A dictionary of query ID to (document ID, graded relevance).</returns>
    public static async Task<IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>>> LoadQrelsAsync(string path, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(path);

        if (!File.Exists(path))
            throw new FileNotFoundException($"Qrels file not found: {path}", path);

        var qrels = new Dictionary<string, Dictionary<string, int>>(StringComparer.Ordinal);

        await using var stream = File.OpenRead(path);
        using var reader = new StreamReader(stream);

        var lineNumber = 0;
        while (true)
        {
            ct.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
            if (line is null)
                break;

            lineNumber++;
            if (lineNumber == 1)
                continue;
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var parts = line.Split('\t');
            if (parts.Length != 3)
                throw new InvalidDataException($"Invalid qrels line {lineNumber}: expected 3 tab-separated columns.");

            var queryId = parts[0];
            var docId = parts[1];
            if (!int.TryParse(parts[2], out var score))
                throw new InvalidDataException($"Invalid qrels score at line {lineNumber}: '{parts[2]}'.");

            if (!qrels.TryGetValue(queryId, out var docs))
            {
                docs = new Dictionary<string, int>(StringComparer.Ordinal);
                qrels[queryId] = docs;
            }

            docs[docId] = score;
        }

        var readOnlyQrels = new Dictionary<string, IReadOnlyDictionary<string, int>>(qrels.Count, StringComparer.Ordinal);
        foreach (var (queryId, docScores) in qrels)
            readOnlyQrels[queryId] = docScores;

        return readOnlyQrels;
    }
}
