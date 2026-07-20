import torch


def peak_gpu_memory_mb() -> float | None:
    """Return peak CUDA memory allocated since the last reset, in MiB."""
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def reset_peak_gpu_memory() -> None:
    """Reset the CUDA peak-memory counter (no-op without CUDA).

    torch's counter is cumulative since process start; training loops reset it
    each step so the logged per-step peak means what its name says.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
