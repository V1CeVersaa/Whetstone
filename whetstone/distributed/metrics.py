from collections.abc import Mapping


def all_reduce_sum(metrics: Mapping[str, float]) -> dict[str, float]:
    """Sum scalar metrics across all ranks, returning the reduced mapping.

    Keys are reduced in sorted order for a consistent tensor layout. Returns the
    metrics unchanged when Torch is unavailable or no process group is active,
    so it is safe to call in single-process mode.
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return dict(metrics)
    if not dist.is_available() or not dist.is_initialized():
        return dict(metrics)
    keys = sorted(metrics)
    tensor = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, tensor.tolist(), strict=True)}
