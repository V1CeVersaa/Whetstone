from dataclasses import dataclass, field
from typing import Any

from whetstone.core.types import BaseModel


@dataclass
class SFTExample(BaseModel):
    """One supervised training pair: a rendered prompt and its target response.

    A lightweight training-side wrapper -- Foundation's ``WhetstoneExample`` /
    ``RenderedPrompt`` remain the source of truth for evaluation. The response
    comes from the dataset's reference solution when available, else from a
    minimal final-answer fallback (see ``train/examples.py``).

    Attributes:
        uid: Originating example's uid.
        source: Originating dataset name.
        prompt_text: Fully rendered prompt (same text the evaluator would use).
        response_text: Supervised target completion, loss-bearing tokens only.
        metadata: Provenance, e.g. ``template_id`` and ``response_source``.
    """

    uid: str
    source: str
    prompt_text: str
    response_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MathRolloutSample(BaseModel):
    """One generated completion inside a grouped RL rollout, with its verdict.

    ``group_id`` is shared by the ``G`` samples generated from the same prompt
    at the same step; group-baseline advantages are computed within it. The
    reward is the binary verifier outcome (Math-RL v0 does no shaping).

    Attributes:
        uid: Originating example's uid.
        group_id: Identity of this sample's rollout group.
        prompt_text: Rendered prompt the completion was sampled from.
        completion_text: Decoded sampled completion.
        reward: ``1.0`` if the math verifier passed, else ``0.0``.
        passed: The verifier's pass flag.
        verifier_reason: Machine-readable verifier outcome, e.g. ``"correct"``.
        extracted_answer: Canonical answer the verifier parsed, if any.
        diagnostics: Verifier diagnostics for debugging.
        num_prompt_tokens: Non-padding prompt token count.
        num_completion_tokens: Generated token count (up to and incl. EOS).
        metadata: Extra provenance (gold answer, finish reason, ...).
    """

    uid: str
    group_id: str
    prompt_text: str
    completion_text: str
    reward: float
    passed: bool
    verifier_reason: str
    extracted_answer: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    num_prompt_tokens: int = 0
    num_completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
