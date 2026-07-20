"""Math grading backed by HuggingFace's `math-verify` package (``math_verify`` v1).

`math-verify <https://github.com/huggingface/Math-Verify>`_ is the grader
behind Open-R1 and lighteval's math leaderboards: LaTeX extraction via
latex2sympy2_extended plus sympy symbolic equivalence (fractions, radicals,
intervals, sets, tuples, symbolic constants). It is Whetstone's sole math
verifier; the judgment authority lives in the package, and this module adapts
it to the :class:`Verifier` protocol so judgments land in the same artifact
schema: a stable reason taxonomy, the ``max_chars`` guard, and diagnostics
recording both this wrapper's version and the installed ``math-verify``
package version (part of run provenance — hold it fixed within a comparison).

The import is deferred to construction so a missing package fails loudly once
at build time instead of per sample.
"""

from importlib import metadata
from typing import Any

from whetstone.core.types import ModelCompletion, VerificationResult, WhetstoneExample


class MathVerifyVerifier:
    """Exact-or-symbolic math answer verifier delegating to ``math-verify``."""

    name = "math_verify"
    version = "v1"

    def __init__(self, *, max_chars: int = 20000) -> None:
        from math_verify import parse, verify

        self.max_chars = max_chars
        self._parse = parse
        self._verify = verify
        try:
            self._package_version = metadata.version("math-verify")
        except metadata.PackageNotFoundError:  # pragma: no cover
            self._package_version = "unknown"

    def verify(
        self,
        example: WhetstoneExample,
        completion: ModelCompletion,
    ) -> VerificationResult:
        """Grade a completion's final answer against the example's gold answer.

        Reason taxonomy (stable across verifier versions so metrics stay
        comparable): ``empty_completion`` / ``too_long`` / ``no_answer_found``
        / ``no_gold_answer`` / ``gold_parse_error`` / ``wrong_answer`` /
        ``correct``, plus ``verifier_error`` when the underlying library
        raises unexpectedly (counted, never crashes the run).
        """
        uid = example.uid
        text = completion.completion or ""

        if not text.strip():
            return self._result(uid, reason="empty_completion")
        if len(text) > self.max_chars:
            return self._result(uid, reason="too_long")
        return self._grade(uid, text, example.final_answer)

    def _grade(self, uid: str, text: str, gold_answer: str | None) -> VerificationResult:
        """Parse and compare via math-verify; any library exception becomes
        ``verifier_error`` (counted in artifacts, never crashes the run)."""
        extracted: list[str] | None = None
        try:
            answer_candidates = self._parse(text)
            if not answer_candidates:
                return self._result(uid, reason="no_answer_found")
            extracted = candidate_strings(answer_candidates)

            if not gold_answer:
                return self._result(uid, reason="no_gold_answer", extracted=extracted)
            gold_text = str(gold_answer)
            if "$" in gold_text:
                gold_candidates = self._parse(gold_text)
            else:
                # Bare gold (no math delimiters) must be wrapped so the latex
                # extractor owns the WHOLE string — raw-text parsing lets
                # expression extraction grab a fragment ("11" out of
                # "11 \sqrt{3}") and silently mis-grade; measured on OpenR1,
                # that false-mismatched ~18% of all targets. Same convention
                # as Open-R1's reward function. Raw parse stays as fallback.
                gold_candidates = self._parse(f"${gold_text}$") or self._parse(gold_text)
            if not gold_candidates:
                return self._result(uid, reason="gold_parse_error", extracted=extracted)

            # math-verify's equivalence is asymmetric for sets/intervals:
            # gold comes first by contract.
            passed = bool(self._verify(gold_candidates, answer_candidates))
        except Exception as exc:
            return self._result(uid, reason="verifier_error", extracted=extracted, error=repr(exc))

        return self._result(
            uid,
            reason="correct" if passed else "wrong_answer",
            passed=passed,
            extracted=extracted,
            gold=candidate_strings(gold_candidates),
        )

    def _result(
        self,
        uid: str,
        *,
        reason: str,
        passed: bool = False,
        extracted: list[str] | None = None,
        gold: list[str] | None = None,
        error: str | None = None,
    ) -> VerificationResult:
        diagnostics: dict[str, Any] = {
            "verifier": {
                "name": self.name,
                "version": self.version,
                "math_verify_package": self._package_version,
            },
            "candidates": extracted or [],
        }
        if gold is not None:
            diagnostics["gold_candidates"] = gold
        if error is not None:
            diagnostics["error"] = error
        return VerificationResult(
            uid=uid,
            domain="math",
            passed=passed,
            reward=1.0 if passed else 0.0,
            score=1.0 if passed else 0.0,
            reason=reason,
            extracted_answer=extracted[0] if extracted else None,
            diagnostics=diagnostics,
        )


def candidate_strings(candidates: list[Any]) -> list[str]:
    """Render math-verify parse results (sympy objects / strings) as strings."""
    return [str(candidate) for candidate in candidates]
