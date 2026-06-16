import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Literal

from whetstone.core.types import ModelCompletion, VerificationResult, WhetstoneExample

AnswerReason = Literal[
    "parsed",
    "empty_completion",
    "too_long",
    "no_answer_found",
    "parse_error",
    "unsupported_expression",
]


@dataclass(frozen=True)
class ParsedAnswer:
    """A successfully parsed answer: its raw text, canonical string, and exact value.

    ``value`` is a :class:`~fractions.Fraction` so two answers compare equal iff
    they are numerically identical (e.g. ``"3/4" == "0.75"``).
    """

    raw: str
    canonical: str
    value: Fraction


@dataclass(frozen=True)
class AnswerExtraction:
    """The outcome of extracting a final answer from completion text.

    ``reason`` is ``"parsed"`` on success (with the value fields populated) or a
    failure code otherwise. ``pattern`` and ``candidates`` record which marker
    matched and the raw candidate strings, for debugging.
    """

    reason: AnswerReason
    raw_answer: str | None = None
    canonical: str | None = None
    value: Fraction | None = None
    pattern: str | None = None
    candidates: tuple[str, ...] = ()
    had_conflict: bool = False


class MathAnswerVerifier:
    """Deterministic exact-match verifier for math final answers (``v1``).

    Extracts a final answer by marker priority, normalizes it to an exact
    rational value, and compares against the gold answer. It deliberately does
    not attempt symbolic equivalence; unsupported forms are reported honestly
    rather than guessed.
    """

    name = "math_answer"
    version = "v1"

    def __init__(self, *, max_chars: int = 20000) -> None:
        self.max_chars = max_chars

    def verify(
        self,
        example: WhetstoneExample,
        completion: ModelCompletion,
    ) -> VerificationResult:
        """Verify a completion's final answer against the example's gold answer."""
        return verify_math_completion(
            uid=example.uid,
            completion=completion.completion,
            gold_answer=example.final_answer,
            max_chars=self.max_chars,
        )


def verify_math_completion(
    *,
    uid: str,
    completion: str,
    gold_answer: str | None,
    max_chars: int = 20000,
) -> VerificationResult:
    """Extract, parse, and grade a completion's final answer against ``gold_answer``.

    Returns a :class:`VerificationResult` whose ``reason`` distinguishes
    extraction failures (``no_answer_found``, ``parse_error``, ...) from a
    parsed-but-wrong answer (``wrong_answer``) and a correct one (``correct``).
    Missing or unparseable gold answers are reported as ``no_gold_answer`` /
    ``gold_parse_error`` rather than counted as model errors.
    """
    extraction: AnswerExtraction = extract_final_answer(completion, max_chars=max_chars)
    diagnostics = {
        "verifier": {"name": MathAnswerVerifier.name, "version": MathAnswerVerifier.version},
        "raw_answer": extraction.raw_answer,
        "pattern": extraction.pattern,
        "candidates": list(extraction.candidates),
        "had_conflict": extraction.had_conflict,
    }

    if extraction.reason != "parsed":
        return VerificationResult(
            uid=uid,
            domain="math",
            passed=False,
            reward=0.0,
            score=0.0,
            reason=extraction.reason,
            extracted_answer=extraction.canonical,
            diagnostics=diagnostics,
        )

    if not gold_answer:
        return VerificationResult(
            uid=uid,
            domain="math",
            passed=False,
            reward=0.0,
            score=0.0,
            reason="no_gold_answer",
            extracted_answer=extraction.canonical,
            diagnostics=diagnostics,
        )

    try:
        gold = parse_answer_value(gold_answer)
    except AnswerParseError as exc:
        diagnostics["gold_parse_error"] = str(exc)
        return VerificationResult(
            uid=uid,
            domain="math",
            passed=False,
            reward=0.0,
            score=0.0,
            reason="gold_parse_error",
            extracted_answer=extraction.canonical,
            diagnostics=diagnostics,
        )

    passed = extraction.value == gold.value
    return VerificationResult(
        uid=uid,
        domain="math",
        passed=passed,
        reward=1.0 if passed else 0.0,
        score=1.0 if passed else 0.0,
        reason="correct" if passed else "wrong_answer",
        extracted_answer=extraction.canonical,
        diagnostics={**diagnostics, "gold_canonical": gold.canonical},
    )


def extract_final_answer(text: str, *, max_chars: int = 20000) -> AnswerExtraction:
    """Extract a single final answer from completion text by marker priority.

    Markers are tried in a fixed order (``\\boxed{}`` > ``####`` > ``Final
    answer:`` > ``The answer is``) so ``v1`` stays comparable across runs. The
    first marker group with parseable candidates wins; when a group holds several
    distinct values the *last* one is taken (matching CoT traces that box
    intermediate results before the final answer) and ``had_conflict`` is set.
    """
    if not text or not text.strip():
        return AnswerExtraction(reason="empty_completion")
    if len(text) > max_chars:
        return AnswerExtraction(reason="too_long")

    # Keep this priority order stable so verifier_v1 remains comparable across runs.
    candidate_groups = [
        ("boxed", find_boxed_answers(text)),
        ("gsm8k_hash", find_regex_answers(text, r"(?m)^\s*####\s*(.+?)\s*$")),
        ("final_answer", find_regex_answers(text, r"(?im)^\s*Final answer\s*:?\s*(.+?)\s*$")),
        ("the_answer_is", find_regex_answers(text, r"(?im)^\s*The answer is\s*:?\s*(.+?)\s*$")),
    ]

    for pattern, candidates in candidate_groups:
        cleaned = tuple(candidate.strip() for candidate in candidates if candidate.strip())
        if not cleaned:
            continue
        parsed: list[ParsedAnswer] = []
        failures: list[str] = []
        for candidate in cleaned:
            try:
                parsed.append(parse_answer_value(candidate))
            except AnswerParseError as exc:
                failures.append(exc.reason)
        if not parsed:
            reason = (
                "unsupported_expression" if "unsupported_expression" in failures else "parse_error"
            )
            return AnswerExtraction(reason=reason, pattern=pattern, candidates=cleaned)

        had_conflict = len({item.value for item in parsed}) > 1
        selected = parsed[-1]  # return the last one
        return AnswerExtraction(
            reason="parsed",
            raw_answer=selected.raw,
            canonical=selected.canonical,
            value=selected.value,
            pattern=pattern,
            candidates=cleaned,
            had_conflict=had_conflict,
        )
    return AnswerExtraction(reason="no_answer_found")


def find_boxed_answers(text: str) -> list[str]:
    """Return the contents of every ``\\boxed{...}``, honoring nested braces.

    Uses brace-depth matching rather than a regex so nested constructs like
    ``\\boxed{\\frac{3}{4}}`` are captured whole. Unterminated markers are skipped.
    """

    marker = "\\boxed{"
    answers: list[str] = []
    index = 0
    while True:
        start = text.find(marker, index)
        if start == -1:
            return answers
        content_start = start + len(marker)
        depth = 1
        cursor = content_start
        while cursor < len(text) and depth > 0:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            answers.append(text[content_start : cursor - 1])
            index = cursor
        else:
            index = content_start


def find_regex_answers(text: str, pattern: str) -> list[str]:
    """Return group 1 of every match of ``pattern`` in ``text``."""
    return [match.group(1) for match in re.finditer(pattern, text)]


class AnswerParseError(ValueError):
    """Raised when an answer string cannot be parsed into an exact value.

    Carries a structured ``reason`` (``parse_error`` or ``unsupported_expression``)
    and the offending ``raw`` text.
    """

    def __init__(self, reason: AnswerReason, raw: str) -> None:
        super().__init__(f"{reason}: {raw!r}")
        self.reason = reason
        self.raw = raw


def parse_answer_value(raw: str) -> ParsedAnswer:
    """Parse a cleaned answer string into an exact rational :class:`ParsedAnswer`.

    Accepts integers, decimals, simple ``a/b`` fractions, and ``\\frac{a}{b}``.
    Percentages and other symbolic forms raise :class:`AnswerParseError` with
    reason ``unsupported_expression``; never uses ``eval``.
    """
    cleaned = clean_answer_text(raw)
    if not cleaned:
        raise AnswerParseError("parse_error", raw)
    if "%" in cleaned:
        raise AnswerParseError("unsupported_expression", raw)

    # Foundation v1 only accepts deterministic numeric forms, not symbolic equivalence.
    latex_fraction = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", cleaned)
    if latex_fraction:
        numerator = parse_decimal_or_integer(latex_fraction.group(1), raw)
        denominator = parse_decimal_or_integer(latex_fraction.group(2), raw)
        if denominator == 0:
            raise AnswerParseError("parse_error", raw)
        value = numerator / denominator
        return ParsedAnswer(raw=raw, canonical=canonical_fraction(value), value=value)

    simple_fraction = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)", cleaned)
    if simple_fraction:
        numerator = parse_decimal_or_integer(simple_fraction.group(1), raw)
        denominator = parse_decimal_or_integer(simple_fraction.group(2), raw)
        if denominator == 0:
            raise AnswerParseError("parse_error", raw)
        value = numerator / denominator
        return ParsedAnswer(raw=raw, canonical=canonical_fraction(value), value=value)

    value = parse_decimal_or_integer(cleaned, raw)
    return ParsedAnswer(raw=raw, canonical=canonical_fraction(value), value=value)


def clean_answer_text(raw: str) -> str:
    """Strip currency symbols, thousands separators, LaTeX spaces, and trailing dots."""
    text = str(raw).strip()
    text = re.sub(r"^[$€£¥]\s*", "", text)
    text = re.sub(r"\s*[$€£¥]$", "", text)
    text = text.strip()
    if text.startswith("$") and text.endswith("$") and len(text) > 1:
        text = text[1:-1].strip()
    text = text.replace("\\,", "")
    text = text.replace(",", "")
    text = text.rstrip(".")
    text = text.strip()
    return text


def parse_decimal_or_integer(text: str, raw: str) -> Fraction:
    """Parse an integer or decimal into an exact ``Fraction``.

    Args:
        text: The component to parse (e.g. a fraction numerator).
        raw: The original answer string, used for error reporting.

    Raises:
        AnswerParseError: If ``text`` is not a plain integer/decimal.
    """
    cleaned = clean_answer_text(text)
    if not re.fullmatch(r"[+-]?(?:\d+|\d+\.\d+|\.\d+)", cleaned):
        raise AnswerParseError("unsupported_expression", raw)
    try:
        return Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise AnswerParseError("parse_error", raw) from exc


def canonical_fraction(value: Fraction) -> str:
    """Render a ``Fraction`` as ``"n"`` when integral, else ``"n/d"`` in lowest terms."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
