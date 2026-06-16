from whetstone.verify.math_answer import extract_final_answer, verify_math_completion


def test_boxed_integer() -> None:
    result = extract_final_answer(r"The final result is \boxed{42}.")
    assert result.reason == "parsed"
    assert result.canonical == "42"


def test_boxed_negative_integer() -> None:
    result = extract_final_answer(r"The final result is \boxed{-7}.")
    assert result.reason == "parsed"
    assert result.canonical == "-7"


def test_boxed_fraction() -> None:
    result = extract_final_answer(r"We get \boxed{\frac{3}{4}}.")
    assert result.reason == "parsed"
    assert result.canonical == "3/4"


def test_decimal_answer() -> None:
    result = extract_final_answer("Final answer: -7.5")
    assert result.reason == "parsed"
    assert result.canonical == "-15/2"


def test_comma_number() -> None:
    result = extract_final_answer("#### 1,234")
    assert result.reason == "parsed"
    assert result.canonical == "1234"


def test_gsm8k_hash_answer() -> None:
    result = extract_final_answer("Reasoning...\n#### 15")
    assert result.reason == "parsed"
    assert result.canonical == "15"


def test_the_answer_is_currency_answer() -> None:
    result = extract_final_answer("The answer is $12")
    assert result.reason == "parsed"
    assert result.canonical == "12"


def test_missing_answer() -> None:
    result = extract_final_answer("No final answer is given.")
    assert result.reason == "no_answer_found"


def test_conflicting_boxed_answers_take_last() -> None:
    # CoT traces box intermediate results; the last boxed value is the answer.
    result = extract_final_answer(r"First \boxed{12}, later \boxed{13}.")
    assert result.reason == "parsed"
    assert result.canonical == "13"
    assert result.had_conflict is True


def test_verify_correct_math_answer() -> None:
    result = verify_math_completion(
        uid="u1", completion=r"Therefore \boxed{3/4}.", gold_answer="0.75"
    )
    assert result.passed is True
    assert result.reason == "correct"


def test_verify_wrong_math_answer() -> None:
    result = verify_math_completion(uid="u1", completion=r"Therefore \boxed{7}.", gold_answer="8")
    assert result.passed is False
    assert result.reason == "wrong_answer"
