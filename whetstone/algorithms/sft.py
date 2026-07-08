"""Supervised fine-tuning on math reasoning examples.

``run_sft`` is the config-driven entry point; ``train_sft_loop`` is the pure
training loop, callable with an injected model/tokenizer so integration tests
can run it on a tiny offline model. The loop trains with response-token-only
cross entropy on prompts rendered by the same versioned templates the
Foundation evaluator uses.
"""

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from whetstone.core.seed import set_seed
from whetstone.data import get_dataset_adapter
from whetstone.data.base import DATASET_REGISTRY
from whetstone.distributed.init import single_process_device
from whetstone.eval.runner import write_json, write_status
from whetstone.models.loader import load_causal_lm
from whetstone.models.tokenization import configure_tokenizer_for_training
from whetstone.prompts.templates import TEMPLATE_REGISTRY
from whetstone.train.checkpointing import (
    checkpoint_dir_for_step,
    last_checkpoint_dir,
    save_checkpoint,
)
from whetstone.train.collators import collate_sft_examples
from whetstone.train.config import SFTTrainConfig, TrainingParams, load_sft_config
from whetstone.train.examples import build_sft_examples
from whetstone.train.loop import (
    create_train_run_dir,
    cycle_index_batches,
    enable_gradient_checkpointing,
    run_checkpoint_eval,
    write_text_samples,
    write_train_run_metadata,
)
from whetstone.train.losses import sft_loss_from_logits
from whetstone.train.metrics import MetricsLogger
from whetstone.train.optim import build_lr_scheduler, build_optimizer, clip_gradients
from whetstone.train.types import SFTExample
from whetstone.utils.logging import configure_logging, get_logger
from whetstone.utils.memory import peak_gpu_memory_mb

logger = get_logger(__name__)

# Called after each checkpoint save with (checkpoint_dir, step).
CheckpointHook = Callable[[Path, int], None]


def run_sft(config_path: str | Path | Sequence[str | Path]) -> Path:
    """Run one config-driven SFT training run and return its run directory."""
    config_paths: list[Path] = (
        [Path(config_path)]
        if isinstance(config_path, (str, Path))
        else [Path(path) for path in config_path]
    )
    config: SFTTrainConfig = load_sft_config(config_paths)
    validate_train_references(config.dataset.name, config.prompt.template_id)

    configure_logging(force=True)
    set_seed(config.run.seed)
    device = single_process_device(config.device)

    run_dir = create_train_run_dir(config.run, config_paths[0].stem)
    write_train_run_metadata(run_dir, config, device=device)
    logger.info(
        f"SFT run {run_dir.name} | dataset={config.dataset.name} split={config.dataset.split} "
        f"limit={config.dataset.limit} template={config.prompt.template_id} "
        f"model={config.model.name_or_path} device={device}"
    )
    try:
        adapter_kwargs = {"streaming": True} if config.dataset.streaming else {}
        adapter = get_dataset_adapter(config.dataset.name, **adapter_kwargs)
        examples = adapter.load(split=config.dataset.split, limit=config.dataset.limit)
        sft_examples, stats = build_sft_examples(examples, config.prompt.template_id)
        write_json(run_dir / "preprocessing.json", stats.to_dict())
        if not sft_examples:
            msg = f"No usable SFT examples (skip reasons: {dict(stats.skip_reasons)})"
            raise ValueError(msg)

        model, tokenizer = load_causal_lm(
            name_or_path=str(config.model.name_or_path),
            dtype=config.model.dtype,
            device=device,
            trust_remote_code=config.model.trust_remote_code,
        )
        tokenizer = configure_tokenizer_for_training(tokenizer)

        def eval_checkpoint(checkpoint: Path, step: int) -> None:
            if config.eval is None:
                return
            run_checkpoint_eval(
                config.eval,
                run_dir=run_dir,
                checkpoint_dir=checkpoint,
                step=step,
                seed=config.run.seed,
                device=device,
                dtype=config.model.dtype,
            )

        final_metrics = train_sft_loop(
            model=model,
            tokenizer=tokenizer,
            sft_examples=sft_examples,
            params=config.training,
            run_dir=run_dir,
            device=device,
            seed=config.run.seed,
            on_eval_checkpoint=eval_checkpoint if config.eval is not None else None,
        )
        write_json(run_dir / "final_metrics.json", {**final_metrics, **stats.to_dict()})
        write_samples(run_dir, sft_examples)
        write_status(run_dir, status="completed", stage="training", extra=final_metrics)
        logger.info(
            f"SFT run complete: {run_dir} | loss {final_metrics['first_train_loss']:.4f} -> "
            f"{final_metrics['final_train_loss']:.4f} in {final_metrics['wall_clock_seconds']:.1f}s"
        )
    except Exception as exc:
        logger.error(f"SFT run failed: {type(exc).__name__}: {exc}")
        write_status(
            run_dir,
            status="failed",
            stage="training",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    return run_dir


def train_sft_loop(
    *,
    model: Any,
    tokenizer: Any,
    sft_examples: list[SFTExample],
    params: TrainingParams,
    run_dir: Path,
    device: str = "cpu",
    seed: int = 42,
    on_eval_checkpoint: CheckpointHook | None = None,
) -> dict[str, Any]:
    """Train ``model`` on ``sft_examples`` and write training artifacts.

    Pure with respect to config loading and model construction, so tests can
    drive it with a tiny offline model. Writes ``metrics.jsonl`` rows, periodic
    checkpoints, and a final ``checkpoints/last``; returns summary metrics
    including the first/final logged loss for overfit verification.
    """
    max_steps = resolve_max_steps(params, num_examples=len(sft_examples))
    optimizer = build_optimizer(
        model, learning_rate=params.learning_rate, weight_decay=params.weight_decay
    )
    scheduler = build_lr_scheduler(
        optimizer, name=params.lr_scheduler, warmup_steps=params.warmup_steps
    )
    metrics_logger = MetricsLogger(run_dir / "metrics.jsonl")
    batches = cycle_index_batches(
        len(sft_examples), params.batch_size, shuffle=params.shuffle, seed=seed
    )

    model.train()
    if params.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    first_loss: float | None = None
    last_loss = 0.0
    start_time = perf_counter()
    logger.info(
        f"SFT: {len(sft_examples)} examples, {max_steps} steps, "
        f"batch={params.batch_size} x accum={params.gradient_accumulation_steps}, device={device}"
    )

    for step in range(1, max_steps + 1):
        step_start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        tokens_processed = 0
        examples_processed = 0

        for _ in range(params.gradient_accumulation_steps):
            batch_examples = [sft_examples[i] for i in next(batches)]
            batch = collate_sft_examples(
                batch_examples, tokenizer, max_seq_length=params.max_seq_length
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            loss = sft_loss_from_logits(logits, batch["input_ids"], batch["response_mask"])
            (loss / params.gradient_accumulation_steps).backward()
            loss_sum += float(loss.detach())
            tokens_processed += int(batch["attention_mask"].sum())
            examples_processed += len(batch_examples)

        grad_norm = clip_gradients(model, params.max_grad_norm)
        optimizer.step()
        scheduler.step()

        step_time = perf_counter() - step_start
        train_loss = loss_sum / params.gradient_accumulation_steps
        if first_loss is None:
            first_loss = train_loss
        last_loss = train_loss

        if step % params.log_every == 0 or step == max_steps:
            metrics_logger.log(
                step,
                {
                    "train_loss": train_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "tokens_per_second": tokens_processed / step_time if step_time else 0.0,
                    "examples_per_second": examples_processed / step_time if step_time else 0.0,
                    "step_time": step_time,
                    "peak_gpu_memory_mb": peak_gpu_memory_mb(),
                },
            )
            logger.info(f"step {step}/{max_steps} loss={train_loss:.4f}")

        should_save = params.save_every is not None and step % params.save_every == 0
        should_eval = (
            on_eval_checkpoint is not None
            and params.eval_every is not None
            and step % params.eval_every == 0
        )
        if should_save or should_eval:
            checkpoint = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir_for_step(run_dir, step),
                training_state=training_state(step, max_steps, train_loss),
            )
            if should_eval and on_eval_checkpoint is not None:
                # A broken periodic eval (dataset access, OOM in the eval
                # model, ...) must not destroy hours of training; the
                # checkpoint is already saved and can be evaluated later.
                try:
                    on_eval_checkpoint(checkpoint, step)
                except Exception as exc:
                    logger.error(
                        f"Checkpoint eval at step {step} failed "
                        f"({type(exc).__name__}: {exc}); continuing training"
                    )

    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=last_checkpoint_dir(run_dir),
        training_state=training_state(max_steps, max_steps, last_loss),
    )
    wall_clock = perf_counter() - start_time
    logger.info(
        f"SFT loop finished: {max_steps} steps in {wall_clock:.1f}s "
        f"(loss {(first_loss or 0.0):.4f} -> {last_loss:.4f})"
    )
    return {
        "num_steps": max_steps,
        "num_sft_examples": len(sft_examples),
        "first_train_loss": first_loss,
        "final_train_loss": last_loss,
        "wall_clock_seconds": wall_clock,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(),
    }


def resolve_max_steps(params: TrainingParams, *, num_examples: int) -> int:
    """Return the optimizer-step budget, deriving it from ``num_epochs`` if needed."""
    if params.max_steps is not None:
        return params.max_steps
    examples_per_step = params.batch_size * params.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(num_examples / examples_per_step))
    return steps_per_epoch * int(params.num_epochs or 1)


def training_state(step: int, max_steps: int, train_loss: float) -> dict[str, Any]:
    """Small resumable-state dict stored beside each checkpoint."""
    return {"step": step, "max_steps": max_steps, "train_loss": train_loss}


def validate_train_references(dataset_name: str, template_id: str) -> None:
    """Fail before run-dir creation when the config references unknown ids."""
    problems = []
    if dataset_name.strip().lower() not in DATASET_REGISTRY.names():
        problems.append(
            f"dataset.name={dataset_name!r} (known: {', '.join(DATASET_REGISTRY.names())})"
        )
    if template_id not in TEMPLATE_REGISTRY.names():
        problems.append(
            f"prompt.template_id={template_id!r} (known: {', '.join(TEMPLATE_REGISTRY.names())})"
        )
    if problems:
        msg = "Unknown config reference(s): " + "; ".join(problems)
        raise ValueError(msg)


def write_samples(run_dir: Path, sft_examples: list[SFTExample], *, limit: int = 3) -> None:
    """Write a small ``samples.md`` showing what the model was trained on."""
    sections = [
        (example.uid, f"PROMPT:\n{example.prompt_text}\n\nRESPONSE:\n{example.response_text}")
        for example in sft_examples[:limit]
    ]
    write_text_samples(run_dir / "samples.md", sections)
