from collections.abc import Callable

from whetstone.core.config import VerifierConfig
from whetstone.core.registry import Registry
from whetstone.verify.base import Verifier
from whetstone.verify.code_exec import CodeExecConfig, CodeExecVerifier, verify_code_completion
from whetstone.verify.math_verify import MathVerifyVerifier

# Verifiers have different constructor signatures, so the registry stores a
# factory ``(VerifierConfig) -> Verifier`` per name rather than the class.
VERIFIER_REGISTRY: Registry[Callable[[VerifierConfig], Verifier]] = Registry()


@VERIFIER_REGISTRY.decorator("math_verify")
def _build_math_verify(config: VerifierConfig) -> Verifier:
    return MathVerifyVerifier(max_chars=config.max_chars)


@VERIFIER_REGISTRY.decorator("code_exec")
def _build_code_exec(config: VerifierConfig) -> Verifier:
    return CodeExecVerifier(
        CodeExecConfig(
            timeout_seconds=config.timeout_seconds,
            max_output_bytes=config.max_output_bytes,
            tests_key=config.tests,
            sandbox_backend=config.sandbox_backend,
        )
    )


def build_verifier(config: VerifierConfig) -> Verifier:
    """Construct the verifier named by ``config.name`` via the registry.

    Raises:
        KeyError: If ``config.name`` is not a registered verifier.
    """
    return VERIFIER_REGISTRY.get(config.name)(config)


__all__ = [
    "VERIFIER_REGISTRY",
    "CodeExecVerifier",
    "MathVerifyVerifier",
    "Verifier",
    "build_verifier",
    "verify_code_completion",
]
