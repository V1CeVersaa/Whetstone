import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedState:
    """Snapshot of this process's place in the (optional) distributed run.

    Attributes:
        enabled: True when running under ``torchrun`` (a process group is active).
        rank: Global process rank.
        local_rank: Rank within this node (selects the local GPU).
        world_size: Total number of processes.
        device: Device string this process should use, e.g. ``"cuda:0"``.
    """

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: str

    @property
    def is_main(self) -> bool:
        """True for the rank-0 process, which owns merging and artifact writes."""
        return self.rank == 0


def init_distributed(preferred_device: str = "cuda") -> DistributedState:
    """Detect torchrun and initialize the process group, or return a single-process state.

    When ``RANK`` is absent the run is single-process and no process group is
    created. Otherwise ranks are read from the environment, the local GPU is
    selected, and an NCCL (GPU) or Gloo (CPU) group is initialized.

    Args:
        preferred_device: Device to use in single-process mode (downgraded to
            CPU if CUDA is unavailable).
    """
    if "RANK" not in os.environ:
        return DistributedState(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=single_process_device(preferred_device),
        )

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = f"cuda:{local_rank}"
    else:
        backend = "gloo"
        device = "cpu"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistributedState(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def single_process_device(preferred_device: str) -> str:
    """Return ``preferred_device`` if usable, else ``"cpu"``.

    Downgrades a requested CUDA device to CPU when Torch is missing or no GPU is
    available, so single-process runs work on CPU-only hosts.
    """
    if preferred_device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if preferred_device.startswith("cuda") and torch.cuda.is_available():
        return preferred_device
    return "cpu"


def barrier(state: DistributedState) -> None:
    """Synchronize all ranks; a no-op in single-process mode."""
    if not state.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


def broadcast_object(value: Any, state: DistributedState, *, source: int = 0) -> Any:
    """Broadcast a picklable object from ``source`` to all ranks.

    Returns ``value`` unchanged in single-process mode; otherwise every rank
    receives the source rank's value (used to share the run directory path).
    """
    if not state.enabled:
        return value
    import torch.distributed as dist

    payload = [value if state.rank == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def shutdown_distributed(state: DistributedState) -> None:
    """Destroy the process group when this process initialized one."""
    if not state.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
