using Retrievo.Abstractions;
using Retrievo.Fusion;

namespace Retrievo.Snapshots;

/// <summary>
/// Maps persisted snapshot metadata to concrete fuser implementations.
/// </summary>
internal static class SnapshotFuserRegistry
{
    internal static SnapshotFuserDescriptor Describe(IFuser fuser)
    {
        ArgumentNullException.ThrowIfNull(fuser);

        return fuser switch
        {
            RrfFuser => new SnapshotFuserDescriptor(SnapshotFuserKind.Rrf),
            _ => new SnapshotFuserDescriptor(
                SnapshotFuserKind.Custom,
                fuser.GetType().AssemblyQualifiedName ?? fuser.GetType().FullName ?? fuser.GetType().Name)
        };
    }

    internal static IFuser Resolve(SnapshotFuserDescriptor descriptor, IFuser? overrideFuser = null)
    {
        ArgumentNullException.ThrowIfNull(descriptor);

        if (overrideFuser is not null)
            return overrideFuser;

        return descriptor.Kind switch
        {
            SnapshotFuserKind.Rrf => new RrfFuser(),
            SnapshotFuserKind.Custom => throw new InvalidOperationException(
                $"Snapshot was created with custom fuser '{descriptor.TypeName ?? "unknown"}'. " +
                "Provide the same IFuser instance when importing."),
            _ => throw new NotSupportedException(
                $"Snapshot fuser kind '{descriptor.Kind}' is not supported.")
        };
    }
}

/// <summary>
/// Snapshot-persisted fuser kinds.
/// </summary>
internal enum SnapshotFuserKind
{
    Rrf,
    Custom
}

/// <summary>
/// Persisted metadata describing which fuser should be used when importing a snapshot.
/// </summary>
internal sealed record SnapshotFuserDescriptor(SnapshotFuserKind Kind, string? TypeName = null);
