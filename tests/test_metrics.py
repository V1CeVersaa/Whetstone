from whetstone.eval.metrics import compute_metrics


def test_math_metrics_match_fixture() -> None:
    rows = [
        {
            "domain": "math",
            "passed": True,
            "reward": 1.0,
            "reason": "correct",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 2,
        },
        {
            "domain": "math",
            "passed": True,
            "reward": 1.0,
            "reason": "correct",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 2,
        },
        {
            "domain": "math",
            "passed": False,
            "reward": 0.0,
            "reason": "wrong_answer",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 4,
            "diagnostics": {"had_conflict": True},
        },
        {
            "domain": "math",
            "passed": False,
            "reward": 0.0,
            "reason": "no_answer_found",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 4,
        },
    ]
    metrics = compute_metrics(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["parse_success_rate"] == 0.75
    assert metrics["no_answer_rate"] == 0.25
    assert metrics["conflicting_answer_rate"] == 0.25
    assert metrics["median_completion_tokens"] == 3.0


def test_code_metrics_match_fixture() -> None:
    rows = [
        {
            "domain": "code",
            "passed": True,
            "reward": 1.0,
            "reason": "passed",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 2,
        },
        {
            "domain": "code",
            "passed": False,
            "reward": 0.5,
            "reason": "wrong_answer",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 3,
        },
        {
            "domain": "code",
            "passed": False,
            "reward": 0.0,
            "reason": "runtime_error",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 4,
        },
        {
            "domain": "code",
            "passed": False,
            "reward": 0.0,
            "reason": "timeout",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 5,
        },
        {
            "domain": "code",
            "passed": False,
            "reward": 0.0,
            "reason": "forbidden_import",
            "num_prompt_tokens": 1,
            "num_completion_tokens": 5,
        },
    ]
    metrics = compute_metrics(rows)
    assert metrics["pass_at_1"] == 0.2
    assert metrics["avg_public_test_pass_rate"] == 0.3
    assert metrics["runtime_error_rate"] == 0.2
    assert metrics["timeout_rate"] == 0.2
    assert metrics["forbidden_import_rate"] == 0.2
