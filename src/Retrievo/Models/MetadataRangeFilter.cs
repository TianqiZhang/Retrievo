namespace Retrievo.Models;

/// <summary>
/// A range filter for metadata values. Compares metadata string values using ordinal comparison,
/// which works correctly for ISO 8601 timestamps and zero-padded numeric strings.
/// Both <see cref="Min"/> and <see cref="Max"/> are inclusive. Either can be null for open-ended ranges.
/// </summary>
public sealed record MetadataRangeFilter
{
    /// <summary>
    /// The metadata key to filter on.
    /// </summary>
    public required string Key { get; init; }

    /// <summary>
    /// Inclusive lower bound. Null means no lower bound.
    /// </summary>
    public string? Min { get; init; }

    /// <summary>
    /// Inclusive upper bound. Null means no upper bound.
    /// </summary>
    public string? Max { get; init; }

    /// <summary>
    /// Validates that at least one bound is specified.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when both Min and Max are null.</exception>
    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Key);
        if (Min is null && Max is null)
            throw new ArgumentException("At least one of Min or Max must be specified.", nameof(Min));
    }
}
