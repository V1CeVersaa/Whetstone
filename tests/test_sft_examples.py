import pytest

pytest.importorskip("math_verify")

from whetstone.core.types import WhetstoneExample
from whetstone.data.tiny import TinyMathAdapter
from whetstone.train.examples import build_sft_examples, build_sft_response
from whetstone.train.types import SFTExample


def math_example(**overrides) -> WhetstoneExample:
    fields = {
        "uid": "tiny_math:train:0",
        "domain": "math",
        "source": "tiny_math",
        "split": "train",
        "prompt_raw": "What is 2 + 3?",
        "reference_solution": "2 + 3 = 5, so the answer is 5.",
        "final_answer": "5",
        "metadata": {},
    }
    fields.update(overrides)
    return WhetstoneExample(**fields)


def test_sft_example_construction_from_reference_solution() -> None:
    examples = TinyMathAdapter().load(split="train")
    sft_examples, stats = build_sft_examples(examples, "math_cot_boxed_v1")

    assert stats.num_loaded_examples == len(examples)
    assert stats.num_sft_examples == len(examples)
    assert stats.num_skipped_examples == 0
    # Tiny solutions carry no \boxed marker, so the boxed-target policy
    # canonicalizes every one of them regardless of loose extraction results.
    assert stats.num_target_canonicalized == len(examples)
    assert stats.num_target_verifier_passed_final == len(examples)
    assert stats.target_policy == "ensure_verifiable_boxed_target_v3"
    assert stats.target_verifier_name == "math_verify"
    assert stats.to_dict()["target_verifier_pass_rate"] == 1.0
    assert stats.to_dict()["target_candidate_verifiable_rate"] == 1.0
    assert stats.to_dict()["emitted_target_verifier_pass_rate"] == 1.0
    assert stats.avg_prompt_chars > 0
    assert stats.avg_response_chars > 0

    first = sft_examples[0]
    assert first.uid == examples[0].uid
    assert first.response_text.startswith(str(examples[0].reference_solution))
    assert first.response_text.endswith("Therefore, the final answer is \\boxed{5}.")
    assert examples[0].prompt_raw in first.prompt_text
    assert "\\boxed" in first.prompt_text  # rendered by the versioned template
    assert first.metadata["response_source"] == "reference_solution_canonicalized"
    assert first.metadata["target_was_canonicalized"] is True
    assert first.metadata["template_id"] == "math_cot_boxed_v1"


def test_sft_example_construction_from_final_answer_fallback() -> None:
    example = math_example(reference_solution=None)
    response, source = build_sft_response(example)
    assert response == "The final answer is \\boxed{5}"
    assert source == "final_answer_fallback"

    sft_examples, stats = build_sft_examples([example], "math_cot_boxed_v1")
    assert stats.num_sft_examples == 1
    assert sft_examples[0].response_text == "The final answer is \\boxed{5}"
    assert sft_examples[0].metadata["response_source"] == "final_answer_fallback"
    assert stats.num_target_verifier_passed_initially == 1
    assert stats.num_target_canonicalized == 0


def test_wrong_boxed_answer_is_rejected_not_repaired() -> None:
    # A positively judged mismatch must be dropped; appending the gold would
    # train on contradictory supervision.
    example = math_example(reference_solution="So the result is \\boxed{6}.", final_answer="5")
    sft_examples, stats = build_sft_examples([example], "math_cot_boxed_v1")

    assert sft_examples == []
    assert stats.num_target_answer_mismatches == 1
    assert stats.num_target_declared_conflicts == 1
    assert stats.num_target_canonicalized == 0
    assert stats.skip_reasons["unverifiable_target_wrong_answer"] == 1
    # Dropped targets leave evidence for failed_targets.jsonl.
    (record,) = stats.failed_targets
    assert record["uid"] == example.uid
    assert record["reason"] == "wrong_answer"
    assert record["gold_answer"] == "5"
    assert record["was_canonicalized"] is False
    assert "\\boxed{6}" in record["response_tail"]
    assert stats.to_dict()["target_declared_conflict_rate"] == 1.0


def test_wrong_unboxed_extraction_is_repaired_not_dropped() -> None:
    # Loose extraction judging unboxed prose "wrong" is noise, not a declared
    # contradiction: measured on OpenR1 (a pre-verified dataset), treating it
    # as one falsely rejected ~50% of all traces. Unboxed targets always get
    # the canonical boxed ending; the re-verified result decides acceptance.
    example = math_example(reference_solution="The answer is 6.", final_answer="5")
    sft_examples, stats = build_sft_examples([example], "math_cot_boxed_v1")

    assert len(sft_examples) == 1
    assert stats.num_target_canonicalized == 1
    assert sft_examples[0].response_text.endswith("Therefore, the final answer is \\boxed{5}.")
    assert sft_examples[0].metadata["target_was_canonicalized"] is True


def test_overlong_target_is_rejected_instead_of_extended() -> None:
    example = math_example(reference_solution="x" * 20_001, final_answer="5")
    sft_examples, stats = build_sft_examples([example], "math_cot_boxed_v1")

    assert sft_examples == []
    assert stats.num_target_parse_failures == 1
    assert stats.num_target_canonicalized == 0
    assert stats.skip_reasons["unverifiable_target_too_long"] == 1


def test_verifiability_enforcement_can_be_disabled_explicitly() -> None:
    example = math_example(reference_solution="Unmarked reasoning only.", final_answer="5")
    sft_examples, stats = build_sft_examples(
        [example],
        "math_cot_boxed_v1",
        ensure_verifiable_target=False,
    )

    assert len(sft_examples) == 1
    assert sft_examples[0].response_text == "Unmarked reasoning only."
    assert stats.num_target_parse_failures == 1
    assert stats.num_target_verifier_passed_final == 0
    assert stats.target_policy == "raw_math_target_v1"


def test_examples_without_response_are_skipped_and_counted() -> None:
    unusable = math_example(uid="tiny_math:train:1", reference_solution=None, final_answer=None)
    whitespace_only = math_example(
        uid="tiny_math:train:2", reference_solution="   ", final_answer=""
    )
    usable = math_example(uid="tiny_math:train:3")

    sft_examples, stats = build_sft_examples(
        [unusable, whitespace_only, usable], "math_cot_boxed_v1"
    )
    assert stats.num_loaded_examples == 3
    assert stats.num_sft_examples == 1
    assert stats.num_skipped_examples == 2
    assert stats.skip_reasons["no_reference_solution_or_final_answer"] == 2
    assert [ex.uid for ex in sft_examples] == ["tiny_math:train:3"]


def test_candidate_and_emitted_target_rates_use_distinct_denominators() -> None:
    accepted = math_example(uid="tiny_math:train:accepted")
    rejected = math_example(
        uid="tiny_math:train:rejected",
        reference_solution="The final answer is \\boxed{6}.",
        final_answer="5",
    )

    _, stats = build_sft_examples([accepted, rejected], "math_cot_boxed_v1")
    payload = stats.to_dict()

    assert payload["target_candidate_verifiable_rate"] == 0.5
    assert payload["target_answer_match_rate"] == 0.5
    assert payload["emitted_target_verifier_pass_rate"] == 1.0
    assert payload["target_verifier_pass_rate"] == 1.0


def test_non_math_domains_are_skipped_with_reason() -> None:
    code_example = math_example(uid="tiny_code:train:0", domain="code")
    _, stats = build_sft_examples([code_example], "math_cot_boxed_v1")
    assert stats.num_sft_examples == 0
    assert stats.skip_reasons["unsupported_domain_code"] == 1


def test_sft_example_round_trips_through_dict() -> None:
    original = SFTExample(
        uid="u",
        source="s",
        prompt_text="p",
        response_text="r",
        metadata={"template_id": "math_cot_boxed_v1"},
    )
    restored = SFTExample.from_dict(original.to_dict())
    assert restored == original
