from whetstone.prompts.base import PromptTemplate

MATH_COT_BOXED_V1 = PromptTemplate(
    template_id="math_cot_boxed_v1",
    required_domain="math",
    text=(
        "Question:\n"
        "{question}\n\n"
        "Solve the problem step by step. Put your final answer in the form:\n"
        "\\boxed{{answer}}\n\n"
        "Answer:\n"
    ),
    metadata={"answer_format": "boxed"},
)
