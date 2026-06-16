from whetstone.core.types import WhetstoneExample
from whetstone.prompts.templates import render_prompt


def test_math_prompt_template() -> None:
    example = WhetstoneExample(
        uid="m1",
        domain="math",
        source="fixture",
        split="test",
        prompt_raw="What is 2+2?",
    )
    rendered = render_prompt(example, "math_cot_boxed_v1")
    assert rendered.template_id == "math_cot_boxed_v1"
    assert "Question:" in rendered.text
    assert "\\boxed{answer}" in rendered.text


def test_code_prompt_template() -> None:
    example = WhetstoneExample(
        uid="c1",
        domain="code",
        source="fixture",
        split="test",
        prompt_raw="Echo stdin.",
    )
    rendered = render_prompt(example, "code_python_solution_v1")
    assert rendered.template_id == "code_python_solution_v1"
    assert "Return only code" in rendered.text
