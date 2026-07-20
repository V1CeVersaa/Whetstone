"""Shared scaffolding for training runs and isolated checkpoint evaluation.

Training runs reuse Foundation's artifact conventions (timestamped run dirs,
``env.json``, ``status.json``) rather than inventing a new style. Periodic
evaluation is a thin bridge into the existing Foundation eval runner: training
writes a normal standalone eval config pointing at the checkpoint, but never
loads a second model while the policy and optimizer remain resident. The
staged config is launched later through ``scripts/run_eval.py``.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from whetstone.core.config import RunConfig, save_yaml
from whetstone.core.paths import ensure_dir, resolve_project_path
from whetstone.core.run_id import make_run_id
from whetstone.distributed.init import DistributedState
from whetstone.eval.runner import collect_env, write_json, write_status
from whetstone.train.config import TrainEvalConfig
from whetstone.utils.hash import stable_hash
from whetstone.utils.logging import attach_run_dir_logging, get_logger

logger = get_logger(__name__)


def ensure_single_process_training(runtime_distributed: bool) -> None:
    """Reject accidental torchrun use until the planned DDP path exists.

    Without DDP wrapping and rank-owned artifact writes, multiple training
    processes would update independent policies and race on the same run
    directory. Failing before dataset/model loading keeps Phase 1 honest.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    under_torchrun = "RANK" in os.environ or world_size > 1
    if runtime_distributed or under_torchrun:
        msg = (
            "Phase 1 training is single-process only; do not launch train_sft.py or "
            "train_math_rl.py with torchrun. DDP is deferred to Post.md Phase 2."
        )
        raise RuntimeError(msg)


def validate_train_references(
    dataset_name: str,
    template_id: str,
    *,
    expected_domain: str = "math",
) -> None:
    """Fail before run-dir creation when a config references unknown ids.

    Shared by every training entry point so a typo'd dataset or template name
    is caught before any side effect (run directory, dataset download).
    """
    from whetstone.data import DATASET_REGISTRY, get_dataset_domain
    from whetstone.prompts.templates import TEMPLATE_REGISTRY

    problems = []
    if dataset_name.strip().lower() not in DATASET_REGISTRY.names():
        problems.append(
            f"dataset.name={dataset_name!r} (known: {', '.join(DATASET_REGISTRY.names())})"
        )
    elif get_dataset_domain(dataset_name) != expected_domain:
        problems.append(
            f"dataset.name={dataset_name!r} has domain={get_dataset_domain(dataset_name)!r}, "
            f"expected {expected_domain!r}"
        )
    if template_id not in TEMPLATE_REGISTRY.names():
        problems.append(
            f"prompt.template_id={template_id!r} (known: {', '.join(TEMPLATE_REGISTRY.names())})"
        )
    if problems:
        msg = "Unknown config reference(s): " + "; ".join(problems)
        raise ValueError(msg)


def create_train_run_dir(run: RunConfig, default_name: str) -> Path:
    """Create the run directory for a (single-process) training run.

    Honors an explicit ``run.output_dir``; otherwise builds a timestamped
    directory under ``run.output_root``, exactly like the eval runner. Also
    mirrors all subsequent logging into ``<run_dir>/run.log`` so queued or
    detached runs (pueue, nohup) keep their log beside the artifacts.
    """
    if run.output_dir:
        run_dir = ensure_dir(resolve_project_path(run.output_dir))
    else:
        output_root = resolve_project_path(run.output_root)
        run_dir = ensure_dir(output_root / make_run_id(run.name or default_name))
    attach_run_dir_logging(run_dir)
    return run_dir


def write_train_run_metadata(run_dir: Path, config: Any, *, device: str) -> None:
    """Write ``train_config.yaml``, ``env.json``, and an initial ``status.json``."""
    save_yaml(config, run_dir / "train_config.yaml")
    state = DistributedState(enabled=False, rank=0, local_rank=0, world_size=1, device=device)
    env = collect_env(state)
    env["config_sha256"] = stable_hash(config.model_dump(mode="json"))
    write_json(run_dir / "env.json", env)
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


def stage_checkpoint_eval(
    eval_config: TrainEvalConfig,
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    step: int,
    seed: int,
    device: str,
    dtype: str,
) -> Path:
    """Write a standalone Foundation eval config for a saved checkpoint.

    The eval config is materialized as YAML inside ``eval/step_XXXXXX`` so the
    evaluation is reproducible. It is deliberately not executed here: loading
    an evaluator beside the resident training model and optimizer can OOM and
    couples evaluator failures to training. Returns the config path to launch
    in a separate process after training or on another GPU.
    """
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
    logger.info(
        f"Staged isolated checkpoint eval config: {config_path} | "
        f"launch later with: uv run scripts/run_eval.py --config {config_path}"
    )
    return config_path


def enable_gradient_checkpointing(model: Any) -> None:
    """Turn on activation checkpointing for training forwards, if supported.

    Also disables the KV cache on the model config (incompatible with
    checkpointing during training). The shared generation context temporarily
    restores ``use_cache=True`` for rollout and then restores this setting.
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
    import torch

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
