from collections import Counter
from dataclasses import dataclass, field

from whetstone.core.types import ModelCompletion, VerificationResult, WhetstoneExample
from whetstone.prompts.templates import render_prompt
from whetstone.train.types import SFTExample
from whetstone.utils.logging import get_logger
from whetstone.verify.base import Verifier
from whetstone.verify.math_verify import MathVerifyVerifier

logger = get_logger(__name__)

TARGET_PARSE_FAILURE_REASONS = {
    "empty_completion",
    "too_long",
    "no_answer_found",
    "verifier_error",
}

# Appending a canonical boxed ending cannot fix an overlong target. Nothing
# else disqualifies an *unboxed* trace: math-verify's fallback extraction on
# long prose is noise, not a declaration, so an initial "wrong_answer" on an
# unboxed response must not block repair (OpenR1 traces are pre-verified
# upstream; judging them 50% wrong was a measured false-positive artifact).
# A response that *declares* a boxed answer never reaches canonicalization —
# the structural \boxed check handles the real contradiction case.
TARGET_NON_CANONICALIZABLE_REASONS = {
    "too_long",
}

# Dropped targets are evidence, not just counts; failed_targets.jsonl keeps
# enough of them to diagnose why a funnel narrowed without rerunning anything.
MAX_FAILED_TARGET_RECORDS = 500


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
    target_policy: str = "ensure_verifiable_boxed_target_v3"
    target_verifier_name: str = MathVerifyVerifier.name
    target_verifier_version: str = MathVerifyVerifier.version
    num_target_candidates: int = 0
    num_target_verifier_passed_initially: int = 0
    num_target_parse_failures: int = 0
    num_target_answer_mismatches: int = 0
    num_target_other_failures: int = 0
    num_target_canonicalized: int = 0
    num_target_verifier_passed_final: int = 0
    num_target_declared_conflicts: int = 0
    # Dropped-target records for failed_targets.jsonl (not in to_dict: the
    # counts live in preprocessing.json, the evidence in its own artifact).
    failed_targets: list[dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        candidate_denominator = max(1, self.num_target_candidates)
        emitted_denominator = max(1, self.num_sft_examples)
        candidate_verifiable_rate = self.num_target_verifier_passed_final / candidate_denominator
        emitted_pass_rate = self.num_target_verifier_passed_final / emitted_denominator
        return {
            "num_loaded_examples": self.num_loaded_examples,
            "num_sft_examples": self.num_sft_examples,
            "num_skipped_examples": self.num_skipped_examples,
            "skip_reasons": dict(self.skip_reasons),
            "avg_prompt_chars": self.avg_prompt_chars,
            "avg_response_chars": self.avg_response_chars,
            "target_policy": self.target_policy,
            "target_verifier": {
                "name": self.target_verifier_name,
                "version": self.target_verifier_version,
            },
            "num_target_candidates": self.num_target_candidates,
            "num_target_verifier_passed_initially": self.num_target_verifier_passed_initially,
            "num_target_parse_failures": self.num_target_parse_failures,
            "num_target_answer_mismatches": self.num_target_answer_mismatches,
            "num_target_other_failures": self.num_target_other_failures,
            "num_target_canonicalized": self.num_target_canonicalized,
            "num_target_verifier_passed_final": self.num_target_verifier_passed_final,
            "num_target_declared_conflicts": self.num_target_declared_conflicts,
            "num_target_initial_parse_failures": self.num_target_parse_failures,
            "num_target_initial_answer_mismatches": self.num_target_answer_mismatches,
            "target_parse_failure_rate": self.num_target_parse_failures / candidate_denominator,
            "target_initial_parse_failure_rate": self.num_target_parse_failures
            / candidate_denominator,
            "target_answer_mismatch_rate": self.num_target_answer_mismatches
            / candidate_denominator,
            "target_initial_answer_mismatch_rate": self.num_target_answer_mismatches
            / candidate_denominator,
            "target_candidate_verifiable_rate": candidate_verifiable_rate,
            "target_answer_match_rate": candidate_verifiable_rate,
            "target_declared_conflict_rate": self.num_target_declared_conflicts
            / candidate_denominator,
            "emitted_target_verifier_pass_rate": emitted_pass_rate,
            # Backward-compatible alias. New consumers should use the explicit
            # candidate/emitted fields above instead of inferring a denominator.
            "target_verifier_pass_rate": emitted_pass_rate,
        }


def build_sft_response(example: WhetstoneExample) -> tuple[str | None, str]:
    """Construct the supervised response for one example.

    Returns ``(response_text, source)`` where ``source`` names how the response
    was built (``"reference_solution"`` / ``"final_answer_fallback"``) or, when
    no response can be built, ``(None, skip_reason)``.

    Priority: use ``reference_solution`` when present; otherwise fall back to a
    minimal boxed final answer so the target stays verifiable.
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
    *,
    ensure_verifiable_target: bool = True,
    target_verifier: Verifier | None = None,
) -> tuple[list[SFTExample], SFTPreprocessingStats]:
    """Turn Foundation examples into SFT pairs via the versioned prompt template.

    Prompts are rendered with the exact template the evaluator uses, so the
    trained distribution matches the evaluated one. Examples that cannot
    produce a response are skipped and counted, never silently dropped.

    ``target_verifier`` judges target verifiability (default: ``math_verify``,
    the same verifier the downstream reward/eval uses, so "verifiable at
    training time" means "rewardable at RL time").

    Target policy (``ensure_verifiable_boxed_target_v3``): a response without
    a ``\\boxed{`` marker gets the canonical boxed gold ending appended — even
    when the initial verification judged it wrong, because loose extraction
    over unboxed prose is noise, not a declaration. A response that *declares*
    a boxed answer contradicting gold is dropped, never "repaired" into
    contradictory supervision. Final acceptance is always the re-verified
    result.
    """
    verifier = target_verifier if target_verifier is not None else MathVerifyVerifier()
    stats = SFTPreprocessingStats(
        num_loaded_examples=len(examples),
        target_policy=(
            "ensure_verifiable_boxed_target_v3"
            if ensure_verifiable_target
            else "raw_math_target_v1"
        ),
        target_verifier_name=getattr(verifier, "name", "unknown"),
        target_verifier_version=getattr(verifier, "version", "unknown"),
    )
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

        stats.num_target_candidates += 1
        declared_boxed_answer = "\\boxed{" in response_text
        initial_verification = verify_sft_target(verifier, example, response_text)
        if initial_verification.passed:
            stats.num_target_verifier_passed_initially += 1
        elif initial_verification.reason in TARGET_PARSE_FAILURE_REASONS:
            stats.num_target_parse_failures += 1
        elif initial_verification.reason == "wrong_answer":
            stats.num_target_answer_mismatches += 1
        else:
            stats.num_target_other_failures += 1

        target_was_canonicalized = False
        final_verification = initial_verification
        if (
            ensure_verifiable_target
            and "\\boxed{" not in response_text
            and initial_verification.reason not in TARGET_NON_CANONICALIZABLE_REASONS
            and (example.final_answer or "").strip()
        ):
            response_text = append_canonical_final_answer(response_text, str(example.final_answer))
            response_source = f"{response_source}_canonicalized"
            target_was_canonicalized = True
            stats.num_target_canonicalized += 1
            final_verification = verify_sft_target(verifier, example, response_text)

        if ensure_verifiable_target and not final_verification.passed:
            stats.num_target_declared_conflicts += int(
                declared_boxed_answer and final_verification.reason == "wrong_answer"
            )
            stats.num_skipped_examples += 1
            stats.skip_reasons[f"unverifiable_target_{final_verification.reason}"] += 1
            if len(stats.failed_targets) < MAX_FAILED_TARGET_RECORDS:
                stats.failed_targets.append(
                    {
                        "uid": example.uid,
                        "reason": final_verification.reason,
                        "gold_answer": example.final_answer,
                        "extracted_answer": final_verification.extracted_answer,
                        "was_canonicalized": target_was_canonicalized,
                        "response_source": response_source,
                        "response_tail": response_text[-400:],
                    }
                )
            continue
        if final_verification.passed:
            stats.num_target_verifier_passed_final += 1

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
                    "target_was_canonicalized": target_was_canonicalized,
                    "target_verifier_reason_initial": initial_verification.reason,
                    "target_verifier_reason_final": final_verification.reason,
                },
            )
        )

    stats.num_sft_examples = len(sft_examples)
    if sft_examples:
        stats.avg_prompt_chars = sum(len(ex.prompt_text) for ex in sft_examples) / len(sft_examples)
        stats.avg_response_chars = sum(len(ex.response_text) for ex in sft_examples) / len(
            sft_examples
        )
    skip_note = f" (skip reasons: {dict(stats.skip_reasons)})" if stats.num_skipped_examples else ""
    logger.info(
        f"SFT preprocessing: {stats.num_sft_examples}/{stats.num_loaded_examples} usable, "
        f"{stats.num_skipped_examples} skipped{skip_note}"
    )
    return sft_examples, stats


def verify_sft_target(
    verifier: Verifier,
    example: WhetstoneExample,
    response_text: str,
) -> VerificationResult:
    """Run the production math verifier on one candidate supervised target."""
    return verifier.verify(
        example,
        ModelCompletion(
            uid=example.uid,
            completion=response_text,
            full_text=response_text,
            num_prompt_tokens=0,
            num_completion_tokens=0,
            finish_reason="sft_target",
        ),
    )


def append_canonical_final_answer(response_text: str, final_answer: str) -> str:
    """Append the versioned canonical final-answer marker used by the verifier."""
    return (
        f"{response_text.rstrip()}\n\n"
        f"Therefore, the final answer is \\boxed{{{final_answer.strip()}}}."
    )
