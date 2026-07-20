from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone.core.config import (
    DatasetConfig,
    GenerationConfig,
    ModelConfig,
    PromptConfig,
    RunConfig,
    RuntimeConfig,
    VerifierConfig,
    load_config,
)

_STRICT = ConfigDict(extra="forbid")
PROTECTED_TRAIN_SPLITS = frozenset({"test"})


def validate_math_training_dataset(dataset: DatasetConfig) -> list[str]:
    """Return config problems that would make a math training dataset unsafe."""
    problems: list[str] = []
    split = dataset.split.strip().lower().split("[", maxsplit=1)[0]
    if split in PROTECTED_TRAIN_SPLITS:
        problems.append(f"dataset.split={dataset.split!r} is protected from gradient updates")

    try:
        from whetstone.data import get_dataset_domain

        domain = get_dataset_domain(dataset.name)
    except KeyError:
        # Unknown references are reported by validate_train_references before
        # any run directory or dataset load occurs.
        return problems
    if domain != "math":
        problems.append(
            f"dataset.name={dataset.name!r} declares domain={domain!r}; "
            "math training requires 'math'"
        )
    return problems


class TrainingParams(BaseModel):
    """SFT optimization hyperparameters.

    Attributes:
        max_steps: Optimizer steps to run; derived from ``num_epochs`` if unset.
        num_epochs: Alternative to ``max_steps``; passes over the SFT dataset.
        batch_size: Micro-batch size per forward pass.
        gradient_accumulation_steps: Micro-batches accumulated per optimizer step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient clipping threshold; ``None`` disables clipping.
        lr_scheduler: ``"constant"`` or ``"linear_warmup"`` (warmup then constant).
        warmup_steps: Warmup optimizer steps for ``linear_warmup``.
        max_seq_length: Token cap for prompt+response.
        overlong_policy: What to do with examples whose prompt+response exceed
            ``max_seq_length``: ``"drop"`` (default) skips and counts them --
            a truncated CoT is a noisy target; ``"truncate"`` keeps them,
            trimming the response tail but always preserving a final EOS.
        gradient_checkpointing: Recompute activations in backward to trade
            compute for memory (recommended for long sequences on small GPUs).
        shuffle: Shuffle example order each epoch (seeded, deterministic).
        log_every: Optimizer-step interval for metrics.jsonl rows.
        save_every: Optimizer-step interval for checkpoints; ``None`` = final only.
        eval_every: Optimizer-step interval for staging a standalone Foundation
            eval config; the evaluator is never loaded inside the training process.
    """

    model_config = _STRICT

    max_steps: int | None = Field(default=None, ge=1)
    num_epochs: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1.0e-5, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float | None = Field(default=1.0, gt=0.0)
    lr_scheduler: Literal["constant", "linear_warmup"] = "constant"
    warmup_steps: int = Field(default=0, ge=0)
    max_seq_length: int = Field(default=2048, ge=2)
    overlong_policy: Literal["drop", "truncate"] = "drop"
    gradient_checkpointing: bool = False
    shuffle: bool = True
    log_every: int = Field(default=5, ge=1)
    save_every: int | None = Field(default=None, ge=1)
    eval_every: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _require_step_budget(self) -> Self:
        if self.max_steps is None and self.num_epochs is None:
            msg = "training requires max_steps or num_epochs"
            raise ValueError(msg)
        if self.max_steps is not None and self.num_epochs is not None:
            msg = "training.max_steps and training.num_epochs are alternatives; set only one"
            raise ValueError(msg)
        return self


class SFTPreprocessingConfig(BaseModel):
    """SFT target construction policy.

    ``ensure_verifiable_target`` makes every emitted target pass the
    ``math_verify`` verifier (the same one the RL reward and evals use).
    Unboxed targets get a canonical boxed gold ending appended; targets whose
    *declared* boxed answer contradicts gold are dropped, never repaired.
    ``max_decode_mismatch_rate`` is the allowed fraction of audited examples
    whose separately tokenized prompt/response IDs do not decode back to the
    exact concatenated text. The default is strict because such a mismatch
    changes the supervised sequence before model weights are loaded.
    """

    model_config = _STRICT

    ensure_verifiable_target: bool = True
    max_decode_mismatch_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class TrainEvalConfig(BaseModel):
    """Standalone Foundation evaluation to stage at checkpoint intervals.

    The training loop fills in the model (current checkpoint) and output
    directory. It writes a launchable config but never imports or executes the
    evaluator while the training policy and optimizer remain resident.
    """

    model_config = _STRICT

    dataset: DatasetConfig
    prompt: PromptConfig
    verifier: VerifierConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class SFTTrainConfig(BaseModel):
    """The fully validated description of one SFT training run.

    Reuses Foundation's config blocks (run/dataset/prompt/model) so a training
    config reads exactly like an eval config plus a ``training`` block.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig
    prompt: PromptConfig
    model: ModelConfig
    preprocessing: SFTPreprocessingConfig = Field(default_factory=SFTPreprocessingConfig)
    training: TrainingParams = Field(default_factory=TrainingParams)
    eval: TrainEvalConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @property
    def device(self) -> str:
        """Resolved device: ``runtime.device`` overrides ``model.device``, default cuda."""
        return self.runtime.device or self.model.device or "cuda"

    @model_validator(mode="after")
    def _require_model_name(self) -> Self:
        problems = validate_math_training_dataset(self.dataset)
        if not self.model.name_or_path:
            problems.append("model.name_or_path is required for SFT training")
        if self.training.eval_every is not None and self.eval is None:
            problems.append("training.eval_every requires an eval block to stage")
        if problems:
            raise ValueError("Invalid SFT configuration: " + "; ".join(problems))
        return self


class RLModelConfig(BaseModel):
    """Policy/reference model selection for Math-RL.

    Attributes:
        policy_name_or_path: SFT checkpoint (or hub id) to train; required.
        reference_name_or_path: Reserved for the future detached KL reward-
            shaping implementation. Phase 1 does not load a reference model.
        dtype: Weight dtype keyword.
        device: Target device; ``None`` lets the runner resolve it.
        trust_remote_code: Allow custom modeling code from the repo.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    policy_name_or_path: str
    reference_name_or_path: str | None = None
    dtype: Literal["bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "auto"] = "bf16"
    device: str | None = None
    trust_remote_code: bool = False


class RLParams(BaseModel):
    """Math-RL v0 (REINFORCE with group baseline) hyperparameters.

    Attributes:
        algorithm: Only ``"reinforce_group_baseline"`` is implemented.
        group_size: Completions sampled per prompt (``G``).
        prompts_per_step: Prompts per training step (``B``).
        max_steps: Training steps (one policy update each; rollouts not reused).
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient clipping threshold; ``None`` disables clipping.
        kl_beta: Reserved for future detached KL reward shaping. Phase 1 only
            accepts ``0.0`` because direct differentiation of sampled log-ratios
            does not provide the intended restoring gradient.
        max_seq_length: Token cap for prompt+completion in the update pass.
        update_micro_batch_size: Sequences per update forward pass; the
            ``B * G`` rollout batch is chunked to this size with exact
            gradient accumulation. ``None`` runs the full batch at once --
            with a 150k-vocab model that OOMs small GPUs, so set this
            whenever ``B * G`` is more than a handful of sequences.
        gradient_checkpointing: Recompute activations in the update backward
            pass; generation is unaffected (checkpointing is train-mode only).
        log_every: Step interval for metrics.jsonl rows.
        save_every: Step interval for checkpoints; ``None`` = final only.
        eval_every: Step interval for staging a standalone Foundation eval
            config; the evaluator is never loaded inside the training process.
    """

    model_config = _STRICT

    algorithm: Literal["reinforce_group_baseline"] = "reinforce_group_baseline"
    group_size: int = Field(default=4, ge=2)
    prompts_per_step: int = Field(default=4, ge=1)
    max_steps: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=1.0e-6, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float | None = Field(default=1.0, gt=0.0)
    kl_beta: float = Field(default=0.0, ge=0.0, le=0.0)
    max_seq_length: int = Field(default=2048, ge=2)
    update_micro_batch_size: int | None = Field(default=None, ge=1)
    gradient_checkpointing: bool = False
    log_every: int = Field(default=1, ge=1)
    save_every: int | None = Field(default=None, ge=1)
    eval_every: int | None = Field(default=None, ge=1)


class AdvantageConfig(BaseModel):
    """Group-baseline advantage computation options.

    Attributes:
        type: Only ``"group_mean"`` is implemented.
        normalize: Divide by the in-group reward std (plus ``epsilon``).
        epsilon: Stabilizer for the normalized variant.
    """

    model_config = _STRICT

    type: Literal["group_mean"] = "group_mean"
    normalize: bool = False
    epsilon: float = Field(default=1.0e-6, gt=0.0)


class MathRLConfig(BaseModel):
    """The fully validated description of one Math-RL v0 training run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig
    prompt: PromptConfig
    model: RLModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verifier: VerifierConfig = Field(default_factory=lambda: VerifierConfig(name="math_verify"))
    rl: RLParams = Field(default_factory=RLParams)
    advantage: AdvantageConfig = Field(default_factory=AdvantageConfig)
    eval: TrainEvalConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @property
    def device(self) -> str:
        """Resolved device: ``runtime.device`` overrides ``model.device``, default cuda."""
        return self.runtime.device or self.model.device or "cuda"

    @property
    def reference_path(self) -> str | None:
        """Phase 1 never loads a reference model; KL reward shaping is deferred."""
        return None

    @model_validator(mode="after")
    def _validate_math_rl_contract(self) -> Self:
        problems = validate_math_training_dataset(self.dataset)
        if not self.model.policy_name_or_path.strip():
            problems.append("model.policy_name_or_path must not be empty")
        if self.rl.group_size > 1 and not self.generation.do_sample:
            problems.append("generation.do_sample must be true when rl.group_size > 1")
        if self.generation.max_new_tokens >= self.rl.max_seq_length:
            problems.append(
                "generation.max_new_tokens must be smaller than rl.max_seq_length "
                "so at least one prompt token fits in every update sequence"
            )
        if self.verifier.name != "math_verify":
            problems.append("Math-RL requires verifier.name='math_verify'")
        if self.rl.eval_every is not None and self.eval is None:
            problems.append("rl.eval_every requires an eval block to stage")
        if problems:
            raise ValueError("Invalid Math-RL configuration: " + "; ".join(problems))
        return self


def load_sft_config(
    paths: str | Path | Sequence[str | Path],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> SFTTrainConfig:
    """Load, merge, and validate config file(s) into an :class:`SFTTrainConfig`."""
    return load_config(paths, SFTTrainConfig, overrides=overrides)


def load_math_rl_config(
    paths: str | Path | Sequence[str | Path],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> MathRLConfig:
    """Load, merge, and validate config file(s) into a :class:`MathRLConfig`."""
    return load_config(paths, MathRLConfig, overrides=overrides)
