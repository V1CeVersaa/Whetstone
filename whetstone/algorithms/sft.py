"""Supervised fine-tuning on math reasoning examples.

``run_sft`` is the config-driven entry point; ``train_sft_loop`` is the pure
training loop, callable with an injected model/tokenizer so integration tests
can run it on a tiny offline model. The loop trains with response-token-only
cross entropy on prompts rendered by the same versioned templates the
Foundation evaluator uses.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from whetstone.core.seed import set_seed
from whetstone.data import get_dataset_adapter
from whetstone.distributed.init import single_process_device
from whetstone.eval.runner import write_json, write_status
from whetstone.models.loader import load_causal_lm_model, load_tokenizer
from whetstone.train.checkpointing import (
    checkpoint_dir_for_step,
    last_checkpoint_dir,
    save_checkpoint,
)
from whetstone.train.collators import (
    audit_sft_tokenization,
    collate_sft_examples,
    filter_overlong_sft_examples,
    validate_sft_tokenization_audit,
)
from whetstone.train.config import SFTTrainConfig, TrainingParams, load_sft_config
from whetstone.train.examples import build_sft_examples
from whetstone.train.loop import (
    create_train_run_dir,
    cycle_index_batches,
    enable_gradient_checkpointing,
    ensure_single_process_training,
    stage_checkpoint_eval,
    validate_train_references,
    write_text_samples,
    write_train_run_metadata,
)
from whetstone.train.losses import sft_loss_terms
from whetstone.train.metrics import MetricsLogger
from whetstone.train.optim import build_lr_scheduler, build_optimizer, clip_gradients
from whetstone.train.types import SFTExample
from whetstone.utils.jsonl import write_jsonl
from whetstone.utils.logging import configure_logging, get_logger
from whetstone.utils.memory import peak_gpu_memory_mb, reset_peak_gpu_memory

logger = get_logger(__name__)

# Called after each checkpoint save with (checkpoint_dir, step).
CheckpointHook = Callable[[Path, int], None]


def run_sft(
    config_path: str | Path | Sequence[str | Path],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Run one config-driven SFT training run and return its run directory.

    ``overrides`` (e.g. from CLI ``--set`` flags) deep-merge on top of the
    config files before validation, so the saved ``train_config.yaml`` reflects
    exactly what ran.
    """
    config_paths: list[Path] = (
        [Path(config_path)]
        if isinstance(config_path, str | Path)
        else [Path(path) for path in config_path]
    )
    config: SFTTrainConfig = load_sft_config(config_paths, overrides=overrides)
    validate_train_references(config.dataset.name, config.prompt.template_id)
    ensure_single_process_training(config.runtime.distributed)

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
        sft_examples, stats = build_sft_examples(
            examples,
            config.prompt.template_id,
            ensure_verifiable_target=config.preprocessing.ensure_verifiable_target,
        )
        write_json(run_dir / "preprocessing.json", stats.to_dict())
        if stats.failed_targets:
            # Dropped targets are evidence: keep them inspectable without a rerun.
            write_jsonl(stats.failed_targets, run_dir / "failed_targets.jsonl")
        if not sft_examples:
            msg = f"No usable SFT examples (skip reasons: {dict(stats.skip_reasons)})"
            raise ValueError(msg)

        tokenizer = load_tokenizer(
            str(config.model.name_or_path),
            trust_remote_code=config.model.trust_remote_code,
            for_training=True,
        )
        tokenization_audit = audit_sft_tokenization(
            sft_examples,
            tokenizer,
            max_seq_length=config.training.max_seq_length,
        )
        write_json(run_dir / "tokenization_audit.json", tokenization_audit)
        validate_sft_tokenization_audit(
            tokenization_audit,
            max_decode_mismatch_rate=config.preprocessing.max_decode_mismatch_rate,
        )
        model = load_causal_lm_model(
            name_or_path=str(config.model.name_or_path),
            dtype=config.model.dtype,
            device=device,
            trust_remote_code=config.model.trust_remote_code,
        )

        def eval_checkpoint(checkpoint: Path, step: int) -> None:
            if config.eval is None:
                return
            stage_checkpoint_eval(
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
        write_json(run_dir / "final_metrics.json", {**stats.to_dict(), **final_metrics})
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
    num_overlong_dropped = 0
    if params.overlong_policy == "drop":
        kept_examples, num_overlong_dropped = filter_overlong_sft_examples(
            sft_examples, tokenizer, max_seq_length=params.max_seq_length
        )
        # Preserve one authoritative collection for the caller's samples.md,
        # the training loop, and final metrics. Rebinding a local list would
        # leave run_sft holding the pre-filter examples.
        sft_examples[:] = kept_examples
        if num_overlong_dropped:
            logger.warning(
                f"Dropped {num_overlong_dropped} example(s) over max_seq_length="
                f"{params.max_seq_length} (overlong_policy=drop); {len(sft_examples)} remain"
            )
        if not sft_examples:
            msg = (
                f"All SFT examples exceed max_seq_length={params.max_seq_length}; "
                "raise the limit or set training.overlong_policy: truncate"
            )
            raise ValueError(msg)
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

    run_peak_memory: float | None = None
    for step in range(1, max_steps + 1):
        step_start = perf_counter()
        reset_peak_gpu_memory()
        optimizer.zero_grad(set_to_none=True)
        tokens_processed = 0
        examples_processed = 0

        # Collate the whole step's micro-batches first so the total response-
        # token count is known: each micro-batch then contributes its NLL
        # *sum* normalized by that global count. Equal-weighting per-micro-
        # batch means would up-weight short-response micro-batches per token.
        micro_batches = []
        for _ in range(params.gradient_accumulation_steps):
            batch_examples = [sft_examples[i] for i in next(batches)]
            micro_batches.append(
                collate_sft_examples(
                    batch_examples, tokenizer, max_seq_length=params.max_seq_length
                )
            )
            examples_processed += len(batch_examples)
        total_response_tokens = max(
            1, sum(int(batch["response_mask"].sum()) for batch in micro_batches)
        )

        nll_sum_value = 0.0
        for micro_batch in micro_batches:
            batch = {key: value.to(device) for key, value in micro_batch.items()}
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            nll_sum, _ = sft_loss_terms(logits, batch["input_ids"], batch["response_mask"])
            (nll_sum / total_response_tokens).backward()
            nll_sum_value += float(nll_sum.detach())
            tokens_processed += int(batch["attention_mask"].sum())
            del logits

        grad_norm = clip_gradients(model, params.max_grad_norm)
        optimizer.step()
        scheduler.step()

        step_time = perf_counter() - step_start
        train_loss = nll_sum_value / total_response_tokens
        if first_loss is None:
            first_loss = train_loss
        last_loss = train_loss
        step_peak_memory = peak_gpu_memory_mb()
        if step_peak_memory is not None:
            run_peak_memory = max(run_peak_memory or 0.0, step_peak_memory)

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
                    "peak_gpu_memory_mb": step_peak_memory,
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
                on_eval_checkpoint(checkpoint, step)

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
        "num_training_examples": len(sft_examples),
        "num_overlong_dropped": num_overlong_dropped,
        "first_train_loss": first_loss,
        "final_train_loss": last_loss,
        "wall_clock_seconds": wall_clock,
        "peak_gpu_memory_mb": run_peak_memory,
    }


def resolve_max_steps(params: TrainingParams, *, num_examples: int) -> int:
    """Return the optimizer-step budget, deriving it from ``num_epochs`` if needed."""
    if params.max_steps is not None:
        return params.max_steps
    examples_per_step = params.batch_size * params.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(num_examples / examples_per_step))
    return steps_per_epoch * int(params.num_epochs or 1)


def training_state(step: int, max_steps: int, train_loss: float) -> dict[str, Any]:
    """Inference-checkpoint metadata; this is not resumable trainer state."""
    return {"step": step, "max_steps": max_steps, "train_loss": train_loss}


def write_samples(run_dir: Path, sft_examples: list[SFTExample], *, limit: int = 3) -> None:
    """Write a small ``samples.md`` showing what the model was trained on."""
    sections = [
        (example.uid, f"PROMPT:\n{example.prompt_text}\n\nRESPONSE:\n{example.response_text}")
        for example in sft_examples[:limit]
    ]
    write_text_samples(run_dir / "samples.md", sections)
