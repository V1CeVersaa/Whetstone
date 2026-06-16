from dataclasses import dataclass
from typing import Any

from whetstone.core.types import ModelCompletion, VerificationResult, WhetstoneExample
from whetstone.verify.sandbox import (
    normalize_stdout,
    run_python_subprocess,
    validate_python_program,
)


@dataclass(frozen=True)
class CodeExecConfig:
    """Settings for the code execution verifier.

    Attributes:
        timeout_seconds: Wall-clock limit per test run.
        max_output_bytes: Cap on captured stdout/stderr before flagging excessive output.
        tests_key: Which test group to grade against (e.g. ``"public"``).
        sandbox_backend: Execution backend; only ``"subprocess"`` is implemented.
    """

    timeout_seconds: float = 3.0
    max_output_bytes: int = 20000
    tests_key: str = "public"
    sandbox_backend: str = "subprocess"


class CodeExecVerifier:
    """Executes generated programs against test cases and grades them (``v1``).

    Runs each test through the configured sandbox backend, comparing normalized
    stdout to the expected output. Reward is fractional (``num_passed /
    num_tests``) to give partial credit for future RL.
    """

    name = "code_exec"
    version = "v1"

    def __init__(self, config: CodeExecConfig | None = None) -> None:
        self.config = config or CodeExecConfig()

    def verify(
        self,
        example: WhetstoneExample,
        completion: ModelCompletion,
    ) -> VerificationResult:
        """Execute the completion against the example's tests and return the verdict."""
        return verify_code_completion(
            uid=example.uid,
            completion=completion.completion,
            tests=example.tests,
            config=self.config,
        )


def verify_code_completion(
    *,
    uid: str,
    completion: str,
    tests: dict[str, Any] | list[dict[str, str]] | None,
    config: CodeExecConfig | None = None,
) -> VerificationResult:
    """Validate, run, and grade a candidate program against its test cases.

    Static validation (empty/syntax/forbidden-import) short-circuits before any
    execution. Otherwise *every* test runs so ``reward`` is the true pass
    fraction (``num_passed / num_tests``) independent of test order; ``passed``
    still requires all tests to pass. ``reason`` records the first failure mode
    (``timeout``, ``runtime_error``, ``wrong_answer``, ...).
    """
    config = config or CodeExecConfig()
    diagnostics: dict[str, Any] = {
        "verifier": {"name": CodeExecVerifier.name, "version": CodeExecVerifier.version},
        "timeout_seconds": config.timeout_seconds,
        "max_output_bytes": config.max_output_bytes,
        "sandbox_backend": config.sandbox_backend,
    }
    if config.sandbox_backend != "subprocess":
        return code_result(
            uid,
            passed=False,
            num_tests=0,
            num_passed=0,
            reason="sandbox_error",
            diagnostics={**diagnostics, "error": f"Unsupported backend: {config.sandbox_backend}"},
        )

    selected_tests = select_tests(tests, config.tests_key)
    if not selected_tests:
        return code_result(
            uid,
            passed=False,
            num_tests=0,
            num_passed=0,
            reason="no_tests",
            diagnostics=diagnostics,
        )

    validation_reason = validate_python_program(completion)
    if validation_reason is not None:
        return code_result(
            uid,
            passed=False,
            num_tests=len(selected_tests),
            num_passed=0,
            reason=validation_reason,
            diagnostics=diagnostics,
        )

    num_passed = 0
    first_failed_test_index: int | None = None
    first_failure_reason: str | None = None
    failure_stdout = ""
    failure_stderr = ""
    failure_returncode: int | None = None
    # Run every test: the reward is the true pass fraction, so it must not depend
    # on which test happens to fail first. Only the first failure is recorded for
    # the human-readable reason and the stdout/stderr preview.
    for index, test_case in enumerate(selected_tests):
        result = run_python_subprocess(
            completion,
            stdin=test_case.get("input", ""),
            timeout_seconds=config.timeout_seconds,
            max_output_bytes=config.max_output_bytes,
        )
        if result.timed_out:
            test_reason = "timeout"
        elif result.excessive_output:
            test_reason = "excessive_output"
        elif result.returncode != 0:
            test_reason = "runtime_error"
        elif normalize_stdout(result.stdout) != normalize_stdout(test_case.get("output", "")):
            test_reason = "wrong_answer"
        else:
            num_passed += 1
            continue
        if first_failed_test_index is None:
            first_failed_test_index = index
            first_failure_reason = test_reason
            failure_stdout = result.stdout
            failure_stderr = result.stderr
            failure_returncode = result.returncode

    passed = num_passed == len(selected_tests)
    reason = "passed" if passed else str(first_failure_reason)
    diagnostics.update(
        {
            "num_tests": len(selected_tests),
            "num_passed": num_passed,
            "first_failed_test_index": first_failed_test_index,
            "first_failed_returncode": failure_returncode,
            "stdout_preview": failure_stdout,
            "stderr_preview": failure_stderr,
        }
    )
    return code_result(
        uid,
        passed=passed,
        num_tests=len(selected_tests),
        num_passed=num_passed,
        reason=reason,
        diagnostics=diagnostics,
    )


def select_tests(
    tests: dict[str, Any] | list[dict[str, str]] | None,
    tests_key: str,
) -> list[dict[str, str]]:
    """Pick the test group to grade, falling back to ``public`` then empty.

    Accepts either a bare list of tests or a dict keyed by group; the requested
    ``tests_key`` is preferred, then ``"public"``.
    """
    if tests is None:
        return []
    if isinstance(tests, list):
        return normalize_test_list(tests)
    if tests_key in tests:
        return normalize_test_list(tests[tests_key])
    if "public" in tests:
        return normalize_test_list(tests["public"])
    return []


def normalize_test_list(value: Any) -> list[dict[str, str]]:
    """Coerce a list of test items into uniform ``{"input", "output"}`` string dicts.

    Tolerates ``stdin``/``stdout`` aliases and missing fields (defaulting to
    empty strings); non-list input yields an empty list.
    """
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "input": str(item.get("input") or item.get("stdin") or ""),
                "output": str(item.get("output") or item.get("stdout") or ""),
            }
        )
    return normalized


def code_result(
    uid: str,
    *,
    passed: bool,
    num_tests: int,
    num_passed: int,
    reason: str,
    diagnostics: dict[str, Any],
) -> VerificationResult:
    """Assemble a code :class:`VerificationResult` with fractional pass-rate reward."""
    reward = float(num_passed / num_tests) if num_tests else 0.0
    return VerificationResult(
        uid=uid,
        domain="code",
        passed=passed,
        reward=reward,
        score=reward,
        reason=reason,
        diagnostics={
            **diagnostics,
            "num_tests": num_tests,
            "num_passed": num_passed,
        },
    )
