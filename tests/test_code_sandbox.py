from whetstone.verify.code_exec import CodeExecConfig, verify_code_completion

TESTS = {"public": [{"input": "hello\n", "output": "hello\n"}]}


def test_correct_simple_program() -> None:
    result = verify_code_completion(
        uid="code:1",
        completion="import sys\nprint(sys.stdin.read().strip())\n",
        tests=TESTS,
        config=CodeExecConfig(timeout_seconds=1.0),
    )
    assert result.passed is True
    assert result.reason == "passed"
    assert result.reward == 1.0


def test_syntax_error() -> None:
    result = verify_code_completion(uid="code:2", completion="def bad(:\n", tests=TESTS)
    assert result.reason == "compile_error"


def test_wrong_answer() -> None:
    result = verify_code_completion(uid="code:3", completion="print('nope')\n", tests=TESTS)
    assert result.reason == "wrong_answer"


def test_runtime_error() -> None:
    result = verify_code_completion(
        uid="code:4", completion="raise RuntimeError('x')\n", tests=TESTS
    )
    assert result.reason == "runtime_error"
    assert result.diagnostics["first_failed_returncode"] != 0
    assert "RuntimeError" in result.diagnostics["stderr_preview"]


def test_timeout() -> None:
    result = verify_code_completion(
        uid="code:5",
        completion="while True:\n    pass\n",
        tests=TESTS,
        config=CodeExecConfig(timeout_seconds=0.2),
    )
    assert result.reason == "timeout"


def test_excessive_output() -> None:
    result = verify_code_completion(
        uid="code:6",
        completion="print('x' * 100)\n",
        tests=TESTS,
        config=CodeExecConfig(timeout_seconds=1.0, max_output_bytes=10),
    )
    assert result.reason == "excessive_output"


def test_empty_code() -> None:
    result = verify_code_completion(uid="code:7", completion="", tests=TESTS)
    assert result.reason == "empty_code"


def test_forbidden_import_is_reported_without_execution() -> None:
    result = verify_code_completion(uid="code:8", completion="import os\nprint('x')\n", tests=TESTS)
    assert result.reason == "forbidden_import"
    assert result.diagnostics["num_tests"] == 1


ECHO = "import sys\nprint(sys.stdin.read().strip())\n"


def test_partial_credit_reward_is_order_independent() -> None:
    # An echo program passes tests 0 and 2 but fails the wrong-output test 1.
    # Reward must be the true pass fraction (2/3) regardless of where the
    # failing test sits, and must not stop counting at the first failure.
    passing_a = {"input": "a\n", "output": "a\n"}
    failing = {"input": "b\n", "output": "WRONG\n"}
    passing_c = {"input": "c\n", "output": "c\n"}

    fail_in_middle = verify_code_completion(
        uid="code:partial:1",
        completion=ECHO,
        tests={"public": [passing_a, failing, passing_c]},
        config=CodeExecConfig(timeout_seconds=1.0),
    )
    assert fail_in_middle.passed is False
    assert fail_in_middle.reason == "wrong_answer"
    assert fail_in_middle.diagnostics["num_passed"] == 2
    assert fail_in_middle.diagnostics["num_tests"] == 3
    assert fail_in_middle.reward == 2 / 3

    # Permuting the tests must not change the reward.
    fail_first = verify_code_completion(
        uid="code:partial:2",
        completion=ECHO,
        tests={"public": [failing, passing_a, passing_c]},
        config=CodeExecConfig(timeout_seconds=1.0),
    )
    assert fail_first.reward == fail_in_middle.reward
