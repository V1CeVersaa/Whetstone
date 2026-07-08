from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

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
        max_seq_length: Token cap for prompt+response; longer pairs are truncated.
        gradient_checkpointing: Recompute activations in backward to trade
            compute for memory (recommended for long sequences on small GPUs).
        shuffle: Shuffle example order each epoch (seeded, deterministic).
        log_every: Optimizer-step interval for metrics.jsonl rows.
        save_every: Optimizer-step interval for checkpoints; ``None`` = final only.
        eval_every: Optimizer-step interval for Foundation eval; ``None`` disables.
    """

    model_config = _STRICT

    max_steps: int | None = None
    num_epochs: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    lr_scheduler: Literal["constant", "linear_warmup"] = "constant"
    warmup_steps: int = 0
    max_seq_length: int = 2048
    gradient_checkpointing: bool = False
    shuffle: bool = True
    log_every: int = 5
    save_every: int | None = None
    eval_every: int | None = None

    @model_validator(mode="after")
    def _require_step_budget(self) -> Self:
        if self.max_steps is None and self.num_epochs is None:
            msg = "training requires max_steps or num_epochs"
            raise ValueError(msg)
        return self


class TrainEvalConfig(BaseModel):
    """Periodic Foundation evaluation run during training.

    The training loop fills in the model (current checkpoint) and output
    directory; this block supplies the dataset, prompt, verifier, and decoding.
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
    training: TrainingParams = Field(default_factory=TrainingParams)
    eval: TrainEvalConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @property
    def device(self) -> str:
        """Resolved device: ``runtime.device`` overrides ``model.device``, default cuda."""
        return self.runtime.device or self.model.device or "cuda"

    @model_validator(mode="after")
    def _require_model_name(self) -> Self:
        if not self.model.name_or_path:
            msg = "model.name_or_path is required for SFT training"
            raise ValueError(msg)
        return self


class RLModelConfig(BaseModel):
    """Policy/reference model selection for Math-RL.

    Attributes:
        policy_name_or_path: SFT checkpoint (or hub id) to train; required.
        reference_name_or_path: Frozen KL reference; defaults to the policy path
            when ``kl_beta > 0`` and is not loaded at all when ``kl_beta == 0``.
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
        kl_beta: Weight of the sampled KL-to-reference term; ``0.0`` skips
            loading the reference model entirely.
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
        eval_every: Step interval for Foundation eval; ``None`` disables.
    """

    model_config = _STRICT

    algorithm: Literal["reinforce_group_baseline"] = "reinforce_group_baseline"
    group_size: int = Field(default=4, ge=2)
    prompts_per_step: int = Field(default=4, ge=1)
    max_steps: int = 50
    learning_rate: float = 1.0e-6
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    kl_beta: float = Field(default=0.0, ge=0.0)
    max_seq_length: int = 2048
    update_micro_batch_size: int | None = Field(default=None, ge=1)
    gradient_checkpointing: bool = False
    log_every: int = 1
    save_every: int | None = None
    eval_every: int | None = None


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
    epsilon: float = 1.0e-6


class MathRLConfig(BaseModel):
    """The fully validated description of one Math-RL v0 training run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig
    prompt: PromptConfig
    model: RLModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verifier: VerifierConfig = Field(default_factory=lambda: VerifierConfig(name="math_answer"))
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
        """Reference model path when KL is enabled, else ``None``."""
        if self.rl.kl_beta <= 0.0:
            return None
        return self.model.reference_name_or_path or self.model.policy_name_or_path


def load_sft_config(paths: str | Path | Sequence[str | Path]) -> SFTTrainConfig:
    """Load, merge, and validate config file(s) into an :class:`SFTTrainConfig`."""
    return load_config(paths, SFTTrainConfig)


def load_math_rl_config(paths: str | Path | Sequence[str | Path]) -> MathRLConfig:
    """Load, merge, and validate config file(s) into a :class:`MathRLConfig`."""
    return load_config(paths, MathRLConfig)
