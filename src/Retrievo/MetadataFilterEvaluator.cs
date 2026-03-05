using Retrievo.Models;

namespace Retrievo;

/// <summary>
/// Shared metadata filter evaluation logic used by both <see cref="HybridSearchIndex"/>
/// and <see cref="MutableHybridSearchIndex"/>. Supports exact-match, range, and
/// <see cref="FieldType.StringArray"/> contains-match filters.
/// </summary>
internal static class MetadataFilterEvaluator
{
    /// <summary>
    /// Returns <c>true</c> when the document's metadata satisfies every active filter on the query.
    /// <para>
    /// <see cref="HybridQuery.MetadataFilters"/>: For each key-value pair, if the key is declared
    /// <see cref="FieldType.StringArray"/> in <paramref name="fieldDefinitions"/>, the stored value
    /// is split by the field's delimiter and the filter passes when any element equals the filter value.
    /// Otherwise, an ordinal exact-match is performed.
    /// </para>
    /// <para>
    /// <see cref="HybridQuery.MetadataRangeFilters"/>: Ordinal string comparison (unchanged).
    /// </para>
    /// </summary>
    internal static bool MatchesAll(
        IReadOnlyDictionary<string, string> metadata,
        HybridQuery query,
        IReadOnlyDictionary<string, FieldDefinition> fieldDefinitions)
    {
        // Metadata filters (exact-match or contains-match depending on field definition)
        if (query.MetadataFilters is not null)
        {
            foreach (var (key, value) in query.MetadataFilters)
            {
                if (!metadata.TryGetValue(key, out var docValue) || docValue is null)
                    return false;

                if (value is null)
                    return false;

                if (fieldDefinitions.TryGetValue(key, out var fieldDef) && fieldDef.Type == FieldType.StringArray)
                {
                    // Contains-match: zero-alloc span scan over delimiter-separated segments
                    bool found = false;
                    var span = docValue.AsSpan();
                    var filter = value.AsSpan();

                    while (span.Length > 0)
                    {
                        int idx = span.IndexOf(fieldDef.Delimiter);
                        ReadOnlySpan<char> segment;
                        if (idx < 0)
                        {
                            segment = span;
                            span = ReadOnlySpan<char>.Empty;
                        }
                        else
                        {
                            segment = span[..idx];
                            span = span[(idx + 1)..];
                        }

                        if (segment.Length > 0 && segment.SequenceEqual(filter))
                        {
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                        return false;
                }
                else
                {
                    // Exact ordinal match (default for undeclared fields or FieldType.String)
                    if (!string.Equals(docValue, value, StringComparison.Ordinal))
                        return false;
                }
            }
        }

        // Range filters (ordinal string comparison — works for ISO 8601 and zero-padded numbers)
        if (query.MetadataRangeFilters is not null)
        {
            foreach (var filter in query.MetadataRangeFilters)
            {
                if (!metadata.TryGetValue(filter.Key, out var docValue))
                    return false;

                if (filter.Min is not null && string.Compare(docValue, filter.Min, StringComparison.Ordinal) < 0)
                    return false;

                if (filter.Max is not null && string.Compare(docValue, filter.Max, StringComparison.Ordinal) > 0)
                    return false;
            }
        }

        return true;
    }
}
