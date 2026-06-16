from typing import Protocol

from whetstone.core.types import ModelCompletion, VerificationResult, WhetstoneExample


class Verifier(Protocol):
    """Structural interface for deterministic verifiers.

    A verifier judges one completion against its example and returns a
    :class:`VerificationResult`. The ``name``/``version`` pair identifies the
    verifier in saved artifacts; bump ``version`` rather than silently changing
    behavior so old results stay interpretable.
    """

    name: str
    version: str

    def verify(
        self,
        example: WhetstoneExample,
        completion: ModelCompletion,
    ) -> VerificationResult:
        """Judge ``completion`` against ``example`` and return the verdict."""
        ...
