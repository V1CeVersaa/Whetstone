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


def test_math_fewshot_prompt_template() -> None:
    example = WhetstoneExample(
        uid="m2",
        domain="math",
        source="fixture",
        split="test",
        prompt_raw="What is 2+2?",
    )
    rendered = render_prompt(example, "math_cot_boxed_fewshot_v1")
    assert rendered.template_id == "math_cot_boxed_fewshot_v1"
    assert rendered.metadata["num_shots"] == 4
    # Four exemplars, each demonstrating a parseable boxed final answer.
    assert rendered.text.count("\\boxed{") == 5  # 4 exemplars + instruction line
    assert rendered.text.count("Question:") == 5  # 4 exemplars + the real one
    # The real question comes last and the prompt ends ready for completion.
    assert rendered.text.rstrip().endswith("Answer:")
    assert rendered.text.index("What is 2+2?") > rendered.text.rindex("\\boxed{28}")


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
