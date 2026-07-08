from collections import Counter
from dataclasses import dataclass, field

from whetstone.core.types import WhetstoneExample
from whetstone.prompts.templates import render_prompt
from whetstone.train.types import SFTExample
from whetstone.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SFTPreprocessingStats:
    """Counts describing how Foundation examples became SFT pairs.

    Examples are never silently dropped: every skip is counted under a reason.
    """

    num_loaded_examples: int = 0
    num_sft_examples: int = 0
    num_skipped_examples: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)
    avg_prompt_chars: float = 0.0
    avg_response_chars: float = 0.0

    def to_dict(self) -> dict:
        return {
            "num_loaded_examples": self.num_loaded_examples,
            "num_sft_examples": self.num_sft_examples,
            "num_skipped_examples": self.num_skipped_examples,
            "skip_reasons": dict(self.skip_reasons),
            "avg_prompt_chars": self.avg_prompt_chars,
            "avg_response_chars": self.avg_response_chars,
        }


def build_sft_response(example: WhetstoneExample) -> tuple[str | None, str]:
    """Construct the supervised response for one example.

    Returns ``(response_text, source)`` where ``source`` names how the response
    was built (``"reference_solution"`` / ``"final_answer_fallback"``) or, when
    no response can be built, ``(None, skip_reason)``.

    Priority: use ``reference_solution`` when present; otherwise fall back to a
    minimal boxed final answer so the target stays verifiable by ``math_answer``.
    """
    solution = (example.reference_solution or "").strip()
    if solution:
        return solution, "reference_solution"

    answer = (example.final_answer or "").strip()
    if answer:
        return f"The final answer is \\boxed{{{answer}}}", "final_answer_fallback"

    return None, "no_reference_solution_or_final_answer"


def build_sft_examples(
    examples: list[WhetstoneExample],
    template_id: str,
) -> tuple[list[SFTExample], SFTPreprocessingStats]:
    """Turn Foundation examples into SFT pairs via the versioned prompt template.

    Prompts are rendered with the exact template the evaluator uses, so the
    trained distribution matches the evaluated one. Examples that cannot
    produce a response are skipped and counted, never silently dropped.
    """
    stats = SFTPreprocessingStats(num_loaded_examples=len(examples))
    sft_examples: list[SFTExample] = []

    for example in examples:
        if example.domain != "math":
            stats.num_skipped_examples += 1
            stats.skip_reasons[f"unsupported_domain_{example.domain}"] += 1
            continue

        response_text, response_source = build_sft_response(example)
        if response_text is None:
            stats.num_skipped_examples += 1
            stats.skip_reasons[response_source] += 1
            continue

        rendered = render_prompt(example, template_id)
        sft_examples.append(
            SFTExample(
                uid=example.uid,
                source=example.source,
                prompt_text=rendered.text,
                response_text=response_text,
                metadata={
                    "template_id": template_id,
                    "response_source": response_source,
                    "split": example.split,
                    "final_answer": example.final_answer,
                },
            )
        )

    stats.num_sft_examples = len(sft_examples)
    if sft_examples:
        stats.avg_prompt_chars = sum(len(ex.prompt_text) for ex in sft_examples) / len(
            sft_examples
        )
        stats.avg_response_chars = sum(len(ex.response_text) for ex in sft_examples) / len(
            sft_examples
        )
    skip_note = f" (skip reasons: {dict(stats.skip_reasons)})" if stats.num_skipped_examples else ""
    logger.info(
        f"SFT preprocessing: {stats.num_sft_examples}/{stats.num_loaded_examples} usable, "
        f"{stats.num_skipped_examples} skipped{skip_note}"
    )
    return sft_examples, stats
