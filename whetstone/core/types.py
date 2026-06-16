from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Self


@dataclass
class BaseModel:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict[T: BaseModel](cls: type[T], data: dict[str, Any]) -> T:
        """Construct an instance from a dict, ignoring unknown keys.

        Unknown keys are dropped so the on-disk schema can gain fields without
        breaking reads of older artifacts. Missing optional fields fall back to
        their dataclass defaults; a missing required field raises ``TypeError``.
        """
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        known = {key: value for key, value in data.items() if key in field_names}
        return cls(**known)


@dataclass
class WhetstoneExample(BaseModel):
    """A dataset row normalized into Whetstone's common schema.

    Dataset adapters emit these; everything downstream (prompts, verifiers,
    metrics) consumes them, so per-dataset fields are flattened here and any
    extra provenance is preserved under ``metadata``.

    Attributes:
        uid: Globally unique id, conventionally ``"{source}:{split}:{row_id}"``.
        domain: Either ``"math"`` or ``"code"``; selects prompt and verifier.
        source: Originating dataset name, e.g. ``"gsm8k"``.
        split: Dataset split the row came from.
        prompt_raw: The unrendered problem statement.
        reference_solution: Gold solution text, when the dataset provides one.
        final_answer: Gold final answer for math exact-match verification.
        tests: Code test cases, e.g. ``{"public": [...], "hidden": [...]}``.
        metadata: Dataset-specific provenance that does not fit other fields.
    """

    uid: str
    domain: Literal["math", "code"]
    source: str
    split: str
    prompt_raw: str
    reference_solution: str | None = None
    final_answer: str | None = None
    tests: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderedPrompt(BaseModel):
    """A prompt produced by applying a versioned template to an example.

    Attributes:
        uid: The originating example's uid.
        template_id: Versioned template identity, e.g. ``"math_cot_boxed_v1"``.
        text: The fully rendered prompt text fed to the model.
        metadata: Template-level metadata (version, answer format, ...).
    """

    uid: str
    template_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCompletion(BaseModel):
    """A single model generation plus its token accounting.

    Attributes:
        uid: The originating example's uid.
        completion: Decoded text generated after the prompt.
        full_text: Decoded prompt-plus-completion text.
        num_prompt_tokens: Non-padding prompt token count.
        num_completion_tokens: Non-padding generated token count.
        finish_reason: Why generation stopped, e.g. ``"stop"`` or ``"length"``.
        generation_metadata: Model name, decoding config, and timing.
    """

    uid: str
    completion: str
    full_text: str
    num_prompt_tokens: int
    num_completion_tokens: int
    finish_reason: str | None = None
    generation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult(BaseModel):
    """A verifier's deterministic judgment of one completion.

    Attributes:
        uid: The originating example's uid.
        domain: ``"math"`` or ``"code"``.
        passed: Whether the completion fully satisfied the verifier.
        reward: Training signal in ``[0, 1]`` (fractional for partial code passes).
        score: Currently mirrors ``reward``; kept separate for future metrics.
        reason: Machine-readable outcome, e.g. ``"correct"`` or ``"timeout"``.
        extracted_answer: Canonical answer the verifier parsed, if any.
        diagnostics: Verifier identity and per-check details for debugging.
    """

    uid: str
    domain: Literal["math", "code"]
    passed: bool
    reward: float
    score: float
    reason: str
    extracted_answer: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    """The full provenance of one prediction: example, prompt, completion, verdict.

    This is the unit written to ``predictions.jsonl``. It bundles the four
    pipeline stages so any saved row can be traced back to its dataset row,
    rendered prompt, decoded completion, and verifier decision.
    """

    example: WhetstoneExample
    rendered_prompt: RenderedPrompt
    completion: ModelCompletion
    verification: VerificationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "example": self.example.to_dict(),
            "rendered_prompt": self.rendered_prompt.to_dict(),
            "completion": self.completion.to_dict(),
            "verification": self.verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            example=WhetstoneExample.from_dict(data["example"]),
            rendered_prompt=RenderedPrompt.from_dict(data["rendered_prompt"]),
            completion=ModelCompletion.from_dict(data["completion"]),
            verification=VerificationResult.from_dict(data["verification"]),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten the record into one JSONL row.

        This defines the on-disk ``predictions.jsonl`` schema: a single
        denormalized dict exposing identity, prompt, completion, gold answer,
        verdict, token counts, and nested metadata/diagnostics. Verifiers can
        be re-run against these rows without regenerating completions.
        """
        return {
            "schema_version": "prediction_v1",
            "uid": self.example.uid,
            "domain": self.example.domain,
            "source": self.example.source,
            "split": self.example.split,
            "template_id": self.rendered_prompt.template_id,
            "prompt": self.rendered_prompt.text,
            "prompt_raw": self.example.prompt_raw,
            "completion": self.completion.completion,
            "full_text": self.completion.full_text,
            "reference_solution": self.example.reference_solution,
            "gold_answer": self.example.final_answer,
            "extracted_answer": self.verification.extracted_answer,
            "passed": self.verification.passed,
            "reward": self.verification.reward,
            "score": self.verification.score,
            "reason": self.verification.reason,
            "num_prompt_tokens": self.completion.num_prompt_tokens,
            "num_completion_tokens": self.completion.num_completion_tokens,
            "finish_reason": self.completion.finish_reason,
            "example_metadata": self.example.metadata,
            "prompt_metadata": self.rendered_prompt.metadata,
            "generation_metadata": self.completion.generation_metadata,
            "diagnostics": self.verification.diagnostics,
            "tests": self.example.tests,
        }
