from whetstone.prompts.base import PromptTemplate

CODE_PYTHON_SOLUTION_V1 = PromptTemplate(
    template_id="code_python_solution_v1",
    required_domain="code",
    text=(
        "You are given a programming problem.\n\n"
        "Problem:\n"
        "{problem_statement}\n\n"
        "Write a complete Python 3 program that solves the problem.\n"
        "Return only code. Do not include Markdown fences.\n\n"
        "Code:\n"
    ),
    metadata={"language": "python3"},
)
