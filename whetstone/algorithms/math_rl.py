"""Math-RL v0: REINFORCE with per-prompt group baseline (optional KL-to-ref).

This is deliberately *not* PPO and *not* full GRPO: no critic, no value head,
no reward model, no clipping, no rollout-batch reuse. Each step samples ``G``
completions per prompt, scores them with the existing math verifier (binary
reward), centers rewards by the in-group mean, and takes one REINFORCE step.
The sampled token-level KL term against a frozen reference is an
approximation, logged as such. Correctness and artifact integrity are the
goals of v0, not accuracy gains.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from whetstone.core.seed import set_seed
from whetstone.data import get_dataset_adapter
from whetstone.distributed.init import single_process_device
from whetstone.eval.runner import write_json, write_status
from whetstone.models.loader import load_causal_lm
from whetstone.prompts.templates import render_prompts
from whetstone.rollout.group_advantage import compute_group_advantages
from whetstone.rollout.math_rollout import GenerateFn, generate_grouped_rollouts
from whetstone.train.checkpointing import (
    checkpoint_dir_for_step,
    last_checkpoint_dir,
    save_checkpoint,
)
from whetstone.train.collators import collate_token_sequences
from whetstone.train.config import (
    AdvantageConfig,
    MathRLConfig,
    RLParams,
    load_math_rl_config,
)
from whetstone.train.logprobs import masked_token_logprobs
from whetstone.train.loop import (
    create_train_run_dir,
    cycle_index_batches,
    enable_gradient_checkpointing,
    run_checkpoint_eval,
    write_text_samples,
    write_train_run_metadata,
)
from whetstone.train.losses import math_rl_loss
from whetstone.train.metrics import MetricsLogger, summarize_rollout_step
from whetstone.train.optim import build_lr_scheduler, build_optimizer, clip_gradients
from whetstone.train.types import MathRolloutSample
from whetstone.utils.jsonl import append_jsonl
from whetstone.utils.logging import configure_logging, get_logger
from whetstone.utils.memory import peak_gpu_memory_mb
from whetstone.verify import build_verifier
from whetstone.verify.base import Verifier

logger = get_logger(__name__)

CheckpointHook = Callable[[Path, int], None]


def run_math_rl(config_path: str | Path | Sequence[str | Path]) -> Path:
    """Run one config-driven Math-RL v0 training run and return its run directory."""
    config_paths: list[Path] = (
        [Path(config_path)]
        if isinstance(config_path, (str, Path))
        else [Path(path) for path in config_path]
    )
    config: MathRLConfig = load_math_rl_config(config_paths)

    configure_logging(force=True)
    set_seed(config.run.seed)
    device = single_process_device(config.device)

    run_dir = create_train_run_dir(config.run, config_paths[0].stem)
    write_train_run_metadata(run_dir, config, device=device)
    logger.info(
        f"Math-RL run {run_dir.name} | dataset={config.dataset.name} split={config.dataset.split} "
        f"limit={config.dataset.limit} policy={config.model.policy_name_or_path} "
        f"reference={config.reference_path or '<disabled>'} device={device}"
    )
    try:
        adapter_kwargs = {"streaming": True} if config.dataset.streaming else {}
        adapter = get_dataset_adapter(config.dataset.name, **adapter_kwargs)
        examples = adapter.load(split=config.dataset.split, limit=config.dataset.limit)
        examples = [example for example in examples if example.domain == "math"]
        if not examples:
            msg = f"Dataset {config.dataset.name!r} produced no math examples"
            raise ValueError(msg)
        rendered_prompts = render_prompts(examples, config.prompt.template_id)

        policy, tokenizer = load_causal_lm(
            name_or_path=config.model.policy_name_or_path,
            dtype=config.model.dtype,
            device=device,
            trust_remote_code=config.model.trust_remote_code,
        )
        reference = None
        if config.reference_path is None:
            logger.info("kl_beta=0.0: skipping reference model load")
        else:
            logger.info(f"Loading frozen KL reference from {config.reference_path}")
            reference, _ = load_causal_lm(
                name_or_path=config.reference_path,
                dtype=config.model.dtype,
                device=device,
                trust_remote_code=config.model.trust_remote_code,
            )
            reference.requires_grad_(False)

        verifier = build_verifier(config.verifier)

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

        final_metrics = train_math_rl_loop(
            policy=policy,
            reference=reference,
            tokenizer=tokenizer,
            examples=examples,
            rendered_prompts=rendered_prompts,
            verifier=verifier,
            rl=config.rl,
            advantage=config.advantage,
            generation_config=config.generation.model_dump(),
            run_dir=run_dir,
            device=device,
            seed=config.run.seed,
            on_eval_checkpoint=eval_checkpoint if config.eval is not None else None,
        )
        write_json(run_dir / "final_metrics.json", final_metrics)
        write_status(run_dir, status="completed", stage="training", extra=final_metrics)
        logger.info(
            f"Math-RL run complete: {run_dir} | {final_metrics['num_rollout_samples']} rollout "
            f"samples, mean_reward={final_metrics['mean_reward_overall']:.3f} "
            f"in {final_metrics['wall_clock_seconds']:.1f}s"
        )
    except Exception as exc:
        logger.error(f"Math-RL run failed: {type(exc).__name__}: {exc}")
        write_status(
            run_dir,
            status="failed",
            stage="training",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    return run_dir


def train_math_rl_loop(
    *,
    policy: Any,
    reference: Any | None,
    tokenizer: Any,
    examples: list,
    rendered_prompts: list,
    verifier: Verifier,
    rl: RLParams,
    advantage: AdvantageConfig,
    generation_config: dict[str, Any],
    run_dir: Path,
    device: str = "cpu",
    seed: int = 42,
    generate_fn: GenerateFn | None = None,
    on_eval_checkpoint: CheckpointHook | None = None,
) -> dict[str, Any]:
    """Run the synchronous rollout -> verify -> advantage -> update loop.

    Pure with respect to config loading and model construction, so tests can
    drive it offline with a tiny model and an injected ``generate_fn``. Every
    sampled completion is appended to ``rollout_samples.jsonl`` with its
    reward, verifier reason, and advantage -- RL without raw rollout logs is
    not acceptable.
    """
    if rl.kl_beta > 0.0 and reference is None:
        msg = "rl.kl_beta > 0 requires a reference model"
        raise ValueError(msg)
    if reference is not None:
        reference.eval()
    if rl.gradient_checkpointing:
        enable_gradient_checkpointing(policy)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    optimizer = build_optimizer(
        policy, learning_rate=rl.learning_rate, weight_decay=rl.weight_decay
    )
    scheduler = build_lr_scheduler(optimizer, name="constant")
    metrics_logger = MetricsLogger(run_dir / "metrics.jsonl")
    rollout_path = run_dir / "rollout_samples.jsonl"
    prompt_batches = cycle_index_batches(
        len(examples), rl.prompts_per_step, shuffle=True, seed=seed
    )

    reward_history: list[float] = []
    last_samples: list[MathRolloutSample] = []
    start_time = perf_counter()
    logger.info(
        f"Math-RL: {len(examples)} prompts, {rl.max_steps} steps, "
        f"B={rl.prompts_per_step} x G={rl.group_size}, kl_beta={rl.kl_beta}, device={device}"
    )

    for step in range(1, rl.max_steps + 1):
        rollout_start = perf_counter()
        batch_indices = next(prompt_batches)
        batch_examples = [examples[i] for i in batch_indices]
        batch_prompts = [rendered_prompts[i] for i in batch_indices]

        rollout = generate_grouped_rollouts(
            model=policy,
            tokenizer=tokenizer,
            examples=batch_examples,
            prompts=batch_prompts,
            group_size=rl.group_size,
            generation_config=generation_config,
            verifier=verifier,
            device=device,
            group_prefix=f"step_{step:06d}/",
            generate_fn=generate_fn,
        )
        rewards = [sample.reward for sample in rollout.samples]
        advantages = compute_group_advantages(
            rewards,
            rl.group_size,
            normalize=advantage.normalize,
            epsilon=advantage.epsilon,
        )
        rollout_time = perf_counter() - rollout_start

        update_start = perf_counter()
        sequences = [
            (prompt_ids[-max(1, rl.max_seq_length - len(completion_ids)) :], completion_ids)
            for prompt_ids, completion_ids in zip(
                rollout.prompt_token_ids, rollout.completion_token_ids, strict=True
            )
        ]
        # The update forward is chunked over sequences: B*G long sequences at
        # once materialize a (B*G, seq, 150k-vocab) logits tensor for both the
        # reference and the policy, which OOMs a 24GB GPU. Per-chunk losses are
        # reweighted so the accumulated gradient equals the full-batch one.
        micro = rl.update_micro_batch_size or len(sequences)
        chunks = [
            collate_token_sequences(sequences[i : i + micro], pad_token_id=int(pad_token_id))
            for i in range(0, len(sequences), micro)
        ]
        total_sequences = len(sequences)
        total_response_tokens = max(1, sum(int(c["response_mask"].sum()) for c in chunks))
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        policy_loss_value = 0.0
        kl_value = 0.0
        offset = 0
        for chunk in chunks:
            batch = {key: value.to(device) for key, value in chunk.items()}
            chunk_size = batch["input_ids"].shape[0]
            chunk_advantages = advantages_tensor[offset : offset + chunk_size].to(device)
            offset += chunk_size

            reference_token_logprobs = None
            if rl.kl_beta > 0.0 and reference is not None:
                with torch.no_grad():
                    ref_logits = reference(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                    ).logits
                    reference_token_logprobs = masked_token_logprobs(
                        ref_logits, batch["input_ids"], batch["response_mask"]
                    )
                    del ref_logits

            logits = policy(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            losses = math_rl_loss(
                logits=logits,
                input_ids=batch["input_ids"],
                response_mask=batch["response_mask"],
                advantages=chunk_advantages,
                kl_beta=rl.kl_beta,
                reference_token_logprobs=reference_token_logprobs,
            )
            # math_rl_loss normalizes within the chunk; rescale each term so
            # the sum over chunks matches full-batch normalization exactly.
            policy_weight = chunk_size / total_sequences
            kl_weight = int(batch["response_mask"].sum()) / total_response_tokens
            chunk_loss = losses["policy_loss"] * policy_weight + rl.kl_beta * (
                losses["kl"] * kl_weight
            )
            chunk_loss.backward()
            policy_loss_value += float(losses["policy_loss"].detach()) * policy_weight
            kl_value += float(losses["kl"].detach()) * kl_weight
            del logits

        grad_norm = clip_gradients(policy, rl.max_grad_norm)
        optimizer.step()
        scheduler.step()
        rl_loss_value = policy_loss_value + rl.kl_beta * kl_value
        update_time = perf_counter() - update_start

        for sample, sample_advantage in zip(rollout.samples, advantages, strict=True):
            append_jsonl({"step": step, **sample.to_dict(), "advantage": sample_advantage}, rollout_path)
        reward_history.extend(rewards)
        last_samples = rollout.samples

        if step % rl.log_every == 0 or step == rl.max_steps:
            completion_tokens = sum(sample.num_completion_tokens for sample in rollout.samples)
            row = {
                "rl_loss": rl_loss_value,
                "policy_loss": policy_loss_value,
                "kl": kl_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "rollout_time": rollout_time,
                "update_time": update_time,
                "tokens_per_second": completion_tokens / rollout_time if rollout_time else 0.0,
                "peak_gpu_memory_mb": peak_gpu_memory_mb(),
                **summarize_rollout_step(
                    rollout.samples, advantages, group_size=rl.group_size
                ),
            }
            metrics_logger.log(step, row)
            logger.info(
                f"step {step}/{rl.max_steps} loss={row['rl_loss']:.4f} "
                f"mean_reward={row['mean_reward']:.3f} "
                f"nonzero_group_variance_rate={row['nonzero_group_variance_rate']:.2f}"
            )

        should_save = rl.save_every is not None and step % rl.save_every == 0
        should_eval = (
            on_eval_checkpoint is not None
            and rl.eval_every is not None
            and step % rl.eval_every == 0
        )
        if should_save or should_eval:
            checkpoint = save_checkpoint(
                model=policy,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir_for_step(run_dir, step),
                training_state={"step": step, "max_steps": rl.max_steps},
            )
            if should_eval and on_eval_checkpoint is not None:
                # Periodic eval failures must not kill the RL run; the
                # checkpoint is saved and can be evaluated offline.
                try:
                    on_eval_checkpoint(checkpoint, step)
                except Exception as exc:
                    logger.error(
                        f"Checkpoint eval at step {step} failed "
                        f"({type(exc).__name__}: {exc}); continuing training"
                    )

    save_checkpoint(
        model=policy,
        tokenizer=tokenizer,
        checkpoint_dir=last_checkpoint_dir(run_dir),
        training_state={"step": rl.max_steps, "max_steps": rl.max_steps},
    )
    write_rollout_samples_digest(run_dir, last_samples)
    wall_clock = perf_counter() - start_time
    logger.info(
        f"Math-RL loop finished: {rl.max_steps} steps, {len(reward_history)} rollout samples "
        f"in {wall_clock:.1f}s"
    )
    return {
        "num_steps": rl.max_steps,
        "num_rollout_samples": len(reward_history),
        "mean_reward_overall": sum(reward_history) / len(reward_history)
        if reward_history
        else 0.0,
        "wall_clock_seconds": wall_clock,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(),
    }


def write_rollout_samples_digest(
    run_dir: Path, samples: list[MathRolloutSample], *, limit: int = 3
) -> None:
    """Write ``samples.md`` from the final step's rollouts for quick eyeballing."""
    sections = [
        (
            f"{sample.group_id} (reward={sample.reward}, reason={sample.verifier_reason})",
            f"PROMPT:\n{sample.prompt_text}\n\nCOMPLETION:\n{sample.completion_text}",
        )
        for sample in samples[:limit]
    ]
    write_text_samples(run_dir / "samples.md", sections)
