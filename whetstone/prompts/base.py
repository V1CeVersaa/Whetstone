from dataclasses import dataclass, field
from typing import Any

from whetstone.core.types import RenderedPrompt, WhetstoneExample


@dataclass(frozen=True)
class PromptTemplate:
    """An immutable, versioned prompt template for one domain.

    The ``template_id`` carries an explicit version (e.g. ``..._v1``) and is
    recorded in every prediction, so evaluations stay comparable as templates
    evolve.

    Attributes:
        template_id: Versioned identity recorded in predictions.
        text: A ``str.format`` template; ``{question}`` and ``{problem_statement}``
            are both bound to the example's raw prompt.
        required_domain: Domain this template applies to (``"math"``/``"code"``).
        metadata: Extra template info copied into the rendered prompt.
    """

    template_id: str
    text: str
    required_domain: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, example: WhetstoneExample) -> RenderedPrompt:
        """Render this template for ``example`` into a :class:`RenderedPrompt`.

        Raises:
            ValueError: If the example's domain does not match ``required_domain``.
        """
        if example.domain != self.required_domain:
            msg = (
                f"Template {self.template_id!r} expects domain {self.required_domain!r}, "
                f"got {example.domain!r}"
            )
            raise ValueError(msg)

        rendered = self.text.format(
            question=example.prompt_raw,
            problem_statement=example.prompt_raw,
        )

        return RenderedPrompt(
            uid=example.uid,
            template_id=self.template_id,
            text=rendered,
            metadata={"template_version": self.template_id, **self.metadata},
        )
