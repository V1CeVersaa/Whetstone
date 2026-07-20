"""Tests for the math-verify wrapper (skipped when the package is absent).

The underlying judgments belong to HF's math-verify; these tests pin the
*wrapper* contract: reason taxonomy, guards, diagnostics provenance, and a
few forms that motivated the adoption (irrational answers, sets/intervals)
staying correct end to end.
"""

import pytest

pytest.importorskip("math_verify")

from whetstone.core.config import VerifierConfig
from whetstone.core.types import ModelCompletion, WhetstoneExample
from whetstone.verify import build_verifier
from whetstone.verify.math_verify import MathVerifyVerifier


def math_example(final_answer: str | None) -> WhetstoneExample:
    return WhetstoneExample(
        uid="u1",
        domain="math",
        source="fixture",
        split="test",
        prompt_raw="question",
        final_answer=final_answer,
    )


def completion(text: str) -> ModelCompletion:
    return ModelCompletion(
        uid="u1",
        completion=text,
        full_text=text,
        num_prompt_tokens=0,
        num_completion_tokens=0,
        finish_reason="stop",
    )


def test_registry_builds_math_verify() -> None:
    verifier = build_verifier(VerifierConfig(name="math_verify"))
    assert isinstance(verifier, MathVerifyVerifier)


def test_correct_plain_number() -> None:
    result = MathVerifyVerifier().verify(math_example("42"), completion(r"So \boxed{42}."))
    assert result.passed is True
    assert result.reason == "correct"
    assert result.diagnostics["verifier"]["name"] == "math_verify"
    assert "math_verify_package" in result.diagnostics["verifier"]


def test_symbolic_equivalence() -> None:
    # Irrational answer: needs sympy equivalence, not exact-rational compare.
    result = MathVerifyVerifier().verify(
        math_example(r"\sqrt{2}"), completion(r"The length is \boxed{\sqrt{2}}.")
    )
    assert result.passed is True

    # Fraction/decimal equivalence still holds.
    result = MathVerifyVerifier().verify(math_example("0.75"), completion(r"\boxed{\frac{3}{4}}"))
    assert result.passed is True


def test_bare_latex_gold_is_parsed_whole_not_as_a_fragment() -> None:
    # Without $-wrapping, expression extraction grabs "11" out of "11 \sqrt{3}"
    # and mis-grades a correct answer; measured at ~18% of OpenR1 targets.
    cases = [
        (r"11 \sqrt{3}", r"So the area is \boxed{11 \sqrt{3}}."),
        (r"16^{100}", r"The count is \boxed{16^{100}}."),
        (r"2^{90}-1", r"Total: \boxed{2^{90}-1}."),
    ]
    for gold, text in cases:
        result = MathVerifyVerifier().verify(math_example(gold), completion(text))
        assert result.passed is True, (gold, result.reason, result.extracted_answer)


def test_wrong_answer_reason() -> None:
    result = MathVerifyVerifier().verify(math_example("8"), completion(r"\boxed{7}"))
    assert result.passed is False
    assert result.reason == "wrong_answer"


def test_no_answer_and_empty_and_too_long_guards() -> None:
    verifier = MathVerifyVerifier(max_chars=10)
    assert verifier.verify(math_example("1"), completion("")).reason == "empty_completion"
    assert verifier.verify(math_example("1"), completion("x" * 11)).reason == "too_long"
    result = MathVerifyVerifier().verify(
        math_example("1"), completion("no final answer marker here at all")
    )
    assert result.reason in {"no_answer_found", "wrong_answer"}


def test_missing_gold_answer_is_dataset_issue_not_model_error() -> None:
    result = MathVerifyVerifier().verify(math_example(None), completion(r"\boxed{5}"))
    assert result.passed is False
    assert result.reason == "no_gold_answer"
