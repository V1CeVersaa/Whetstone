from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT = ConfigDict(extra="forbid")


class RunConfig(BaseModel):
    """Run identity and output location."""

    model_config = _STRICT

    name: str | None = None
    seed: int = 42
    output_root: str = "runs"
    output_dir: str | None = None


class DatasetConfig(BaseModel):
    """Which dataset split and how many rows to evaluate.

    Attributes:
        name: Dataset adapter name, e.g. ``"gsm8k"`` / ``"openr1_math"`` / ``"taco_cobalt"``.
        split: Split to load, e.g. ``"train"`` / ``"test"`` / ``"validation"``.
        limit: Max rows to load; ``None`` loads all.
        streaming: If True, stream the split lazily instead of downloading it fully.
    """

    model_config = _STRICT

    name: str
    split: str = "train"
    limit: int | None = None
    streaming: bool = False


class PromptConfig(BaseModel):
    """Which versioned prompt template to render with.

    Attributes:
        template_id: Registered template id, e.g. ``"math_cot_boxed_v1"``.
    """

    model_config = _STRICT

    template_id: str


class ModelConfig(BaseModel):
    """Model identity, dtype, device, and backend selection.

    Attributes:
        name_or_path: HF hub id or local path; required unless ``backend`` is ``"mock"``.
        dtype: Weight dtype keyword (``bf16`` / ``fp16`` / ``fp32`` / ``auto`` and aliases).
        device: Target device (``"cuda"`` / ``"cpu"`` / ``"cuda:0"`` / ``"auto"``);
            ``None`` lets the runner resolve it (see :attr:`EvalConfig.device`).
        trust_remote_code: Allow custom modeling code from the model repo.
        backend: ``"transformers"`` (default) or ``"mock"`` (no-model debug path).
        mock_mode: Stub-completion mode (``"reference"`` / ``"empty"``) when mock.
    """

    model_config = _STRICT

    name_or_path: str | None = None
    dtype: Literal["bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "auto"] = "bf16"
    device: str | None = None
    trust_remote_code: bool = False
    backend: str | None = None
    mock_mode: str = "reference"


class GenerationConfig(BaseModel):
    """Decoding parameters.

    Unlike the other config blocks this allows extra keys, because the whole
    block (minus ``backend``) is forwarded as ``**kwargs`` to ``model.generate``,
    which accepts many parameters we do not enumerate here
    (``repetition_penalty``, ``num_beams``, ``num_return_sequences``, ...).

    Attributes:
        backend: ``"mock"`` routes to the no-model path; otherwise unused here and
            stripped before the block is forwarded to ``model.generate``.
        max_new_tokens: Max tokens to generate per prompt.
        do_sample: Sample (``True``) vs greedy decoding (``False``).
        temperature: Sampling temperature (``>= 0``).
        top_p: Nucleus-sampling probability mass (``0..1``).
    """

    model_config = ConfigDict(extra="allow")

    backend: str | None = None
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class VerifierConfig(BaseModel):
    """Verifier selection and its parameters (math and code share one block).

    Fields apply to whichever verifier ``name`` selects; the rest are ignored.

    Attributes:
        name: Which verifier to use.
        max_chars: [math_answer] Max completion length before reason ``too_long``.
        tests: [code_exec] Which test group to grade against, e.g. ``"public"``.
        timeout_seconds: [code_exec] Per-test wall-clock limit.
        max_output_bytes: [code_exec] Cap on captured stdout/stderr.
        sandbox_backend: [code_exec] Execution backend; only ``"subprocess"`` is implemented.
    """

    model_config = _STRICT

    name: str
    max_chars: int = 20000  # math_answer
    tests: str = "public"  # code_exec
    timeout_seconds: float = 3.0
    max_output_bytes: int = 20000
    sandbox_backend: str = "subprocess"

    @model_validator(mode="after")
    def _check_known_verifier(self) -> Self:
        from whetstone.verify import VERIFIER_REGISTRY  # noqa: PLC0415

        if self.name not in VERIFIER_REGISTRY.names():
            known = ", ".join(VERIFIER_REGISTRY.names()) or "<none>"
            msg = f"Unknown verifier {self.name!r}; known verifiers: {known}"
            raise ValueError(msg)
        return self


class RuntimeConfig(BaseModel):
    """Runtime overrides layered on top of an eval config.

    Attributes:
        distributed: Declarative flag; the actual distributed state is auto-detected
            from torchrun's ``RANK`` env, so this is informational.
        device: Device override that takes precedence over ``model.device``.
        log_every: Logging interval (examples/steps).
        nproc_per_node: Processes per node for torchrun launches (informational).
    """

    model_config = _STRICT

    distributed: bool = False
    device: str | None = None
    log_every: int = 1
    nproc_per_node: int | None = None


class EvalConfig(BaseModel):
    """The fully validated description of one evaluation run.

    Built by deep-merging one or more YAML fragments and validating the result.
    ``protected_namespaces=()`` allows the ``model`` field name. The blocks
    (``model`` / ``dataset`` / ``prompt`` / ``generation`` / ``verifier`` /
    ``runtime``) are the shared vocabulary future trainer configs reuse.

    Attributes:
        run: Run identity and output location.
        dataset: Which dataset split and how many rows.
        prompt: Which versioned prompt template.
        model: Model identity, dtype, device, backend.
        generation: Decoding / rollout parameters.
        verifier: Verifier (reward source) selection and parameters.
        runtime: Runtime overrides (device, distributed hints).
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig
    prompt: PromptConfig
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verifier: VerifierConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @property
    def effective_backend(self) -> str:
        """Resolved generation backend: ``model.backend`` > ``generation.backend`` > transformers."""
        return self.model.backend or self.generation.backend or "transformers"

    @property
    def device(self) -> str:
        """Resolved device: ``runtime.device`` overrides ``model.device``, default cuda."""
        return self.runtime.device or self.model.device or "cuda"

    @model_validator(mode="after")
    def _require_model_name_unless_mock(self) -> Self:
        """Require model name unless the backend is 'mock'."""
        if self.effective_backend != "mock" and not self.model.name_or_path:
            msg = "model.name_or_path is required unless the backend is 'mock'"
            raise ValueError(msg)
        return self


def load_config[T: BaseModel](paths: str | Path | Sequence[str | Path], schema: type[T]) -> T:
    paths = [paths] if isinstance(paths, (str, Path)) else list[str | Path](paths)
    data = merge_config_files(paths)
    return schema.model_validate(data)


def load_eval_config(paths: str | Path | Sequence[str | Path]) -> EvalConfig:
    """Load, merge, and validate config file(s) into an :class:`EvalConfig`."""
    return load_config(paths, EvalConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in YAML config: {path}")

    return data


def save_yaml(data: Mapping[str, Any] | BaseModel, path: str | Path) -> None:
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else dict(data)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_config_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for path in paths:
        merged = deep_merge(merged, load_yaml(path))

    return merged
