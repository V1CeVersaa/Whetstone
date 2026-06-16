import torch


def peak_gpu_memory_mb() -> float | None:
    """Return peak CUDA memory allocated so far, in MiB."""
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
