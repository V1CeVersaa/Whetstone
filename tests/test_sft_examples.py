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
    assert stats.avg_prompt_chars > 0
    assert stats.avg_response_chars > 0

    first = sft_examples[0]
    assert first.uid == examples[0].uid
    assert first.response_text == examples[0].reference_solution
    assert examples[0].prompt_raw in first.prompt_text
    assert "\\boxed" in first.prompt_text  # rendered by the versioned template
    assert first.metadata["response_source"] == "reference_solution"
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
