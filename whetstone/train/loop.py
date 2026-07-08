"""Shared scaffolding for training runs: run dirs, env metadata, periodic eval.

Training runs reuse Foundation's artifact conventions (timestamped run dirs,
``env.json``, ``status.json``) rather than inventing a new style. Periodic
evaluation is a thin bridge into the existing Foundation eval runner: it
writes a normal eval config pointing at the checkpoint and calls
``run_evaluation``, so trained checkpoints are judged by exactly the same
pipeline as any other model.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from whetstone.core.config import RunConfig, save_yaml
from whetstone.core.paths import ensure_dir, resolve_project_path
from whetstone.core.run_id import make_run_id
from whetstone.distributed.init import DistributedState
from whetstone.eval.runner import collect_env, write_json, write_status
from whetstone.train.config import TrainEvalConfig
from whetstone.utils.logging import get_logger

logger = get_logger(__name__)


def create_train_run_dir(run: RunConfig, default_name: str) -> Path:
    """Create the run directory for a (single-process) training run.

    Honors an explicit ``run.output_dir``; otherwise builds a timestamped
    directory under ``run.output_root``, exactly like the eval runner.
    """
    if run.output_dir:
        return ensure_dir(resolve_project_path(run.output_dir))
    output_root = resolve_project_path(run.output_root)
    return ensure_dir(output_root / make_run_id(run.name or default_name))


def write_train_run_metadata(run_dir: Path, config: Any, *, device: str) -> None:
    """Write ``train_config.yaml``, ``env.json``, and an initial ``status.json``."""
    save_yaml(config, run_dir / "train_config.yaml")
    state = DistributedState(
        enabled=False, rank=0, local_rank=0, world_size=1, device=device
    )
    write_json(run_dir / "env.json", collect_env(state))
    write_status(run_dir, status="running", stage="training")


def build_checkpoint_eval_config(
    eval_config: TrainEvalConfig,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    seed: int,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    """Assemble a Foundation eval config dict targeting a local checkpoint."""
    return {
        "run": {
            "name": f"eval_{checkpoint_dir.name}",
            "seed": seed,
            "output_dir": str(output_dir),
        },
        "dataset": eval_config.dataset.model_dump(mode="json"),
        "prompt": eval_config.prompt.model_dump(mode="json"),
        "model": {"name_or_path": str(checkpoint_dir), "dtype": dtype, "device": device},
        "generation": eval_config.generation.model_dump(mode="json"),
        "verifier": eval_config.verifier.model_dump(mode="json"),
        "runtime": {"distributed": False, "device": device},
    }


def run_checkpoint_eval(
    eval_config: TrainEvalConfig,
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    step: int,
    seed: int,
    device: str,
    dtype: str,
) -> Path:
    """Evaluate a saved checkpoint with the Foundation eval runner.

    The eval config is materialized as YAML inside ``eval/step_XXXXXX`` so the
    evaluation is reproducible standalone, then executed in-process. Returns
    the eval run directory.
    """
    from whetstone.eval.runner import run_evaluation  # noqa: PLC0415  (import cycle guard)

    eval_dir = ensure_dir(run_dir / "eval" / f"step_{step:06d}")
    config_dict = build_checkpoint_eval_config(
        eval_config,
        checkpoint_dir=checkpoint_dir,
        output_dir=eval_dir,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    config_path = eval_dir / "eval_config.yaml"
    save_yaml(config_dict, config_path)
    logger.info(f"Evaluating checkpoint {checkpoint_dir} -> {eval_dir}")
    return run_evaluation(config_path)


def enable_gradient_checkpointing(model: Any) -> None:
    """Turn on activation checkpointing for training forwards, if supported.

    Also disables the KV cache on the model config (incompatible with
    checkpointing during training). HF applies checkpointing only when
    ``model.training`` is True, so eval-mode ``generate`` keeps its cache.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info("Gradient checkpointing enabled")
    else:
        logger.warning("Model does not support gradient checkpointing; continuing without it")


def cycle_index_batches(
    num_items: int,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[list[int]]:
    """Yield index batches forever, reshuffling deterministically each epoch.

    Both training loops draw batches from this: SFT micro-batches and RL prompt
    batches. Shuffling is seeded per epoch (``seed + epoch``), so a run is
    reproducible from its config alone. The final short batch of an epoch is
    yielded as-is rather than padded or dropped.
    """
    import torch  # noqa: PLC0415  (keep module importable without torch at doc time)

    epoch = 0
    while True:
        if shuffle:
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(num_items, generator=generator).tolist()
        else:
            order = list(range(num_items))
        for start in range(0, num_items, batch_size):
            yield order[start : start + batch_size]
        epoch += 1


def write_text_samples(path: Path, sections: list[tuple[str, str]]) -> None:
    """Write a small ``samples.md`` digest of (title, body) text sections."""
    lines = ["# Training Samples", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", "```text", body[:2000], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
