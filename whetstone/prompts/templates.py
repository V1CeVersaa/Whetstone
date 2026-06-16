from whetstone.core.registry import Registry
from whetstone.core.types import RenderedPrompt, WhetstoneExample
from whetstone.prompts.base import PromptTemplate
from whetstone.prompts.code import CODE_PYTHON_SOLUTION_V1
from whetstone.prompts.math import MATH_COT_BOXED_V1

# Templates are registered under their own ``template_id`` so a prediction's
# recorded id is exactly the lookup key.
TEMPLATE_REGISTRY: Registry[PromptTemplate] = Registry()
for _template in (MATH_COT_BOXED_V1, CODE_PYTHON_SOLUTION_V1):
    TEMPLATE_REGISTRY.register(_template.template_id, _template)


def get_prompt_template(template_id: str) -> PromptTemplate:
    """Look up a registered prompt template by id.

    Raises:
        KeyError: If ``template_id`` is unknown; the message lists known ids.
    """
    return TEMPLATE_REGISTRY.get(template_id)


def render_prompt(example: WhetstoneExample, template_id: str) -> RenderedPrompt:
    """Render one example with the named template."""
    return get_prompt_template(template_id).render(example)


def render_prompts(examples: list[WhetstoneExample], template_id: str) -> list[RenderedPrompt]:
    """Render a list of examples with the same template, preserving order."""
    return [render_prompt(example, template_id) for example in examples]
