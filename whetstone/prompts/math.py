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

# Hand-written exemplars (deliberately NOT taken from GSM8K or OpenR1, so no
# eval split is contaminated). Each one ends in the exact final-answer form
# the verifier extracts, because base models follow demonstrated patterns far
# more reliably than written instructions.
_FEWSHOT_EXAMPLES = (
    "Question:\n"
    "Tom has 3 boxes of pencils, and each box holds 12 pencils. He gives 8 pencils "
    "to his sister. How many pencils does he have left?\n\n"
    "Answer:\n"
    "Tom starts with 3 * 12 = 36 pencils. After giving 8 away, he has 36 - 8 = 28 "
    "pencils left. The final answer is \\boxed{{28}}.\n\n"
    "Question:\n"
    "A shirt costs $15 and a pair of pants costs $25. Sarah buys 2 shirts and 1 pair "
    "of pants. How much does she spend in total?\n\n"
    "Answer:\n"
    "The shirts cost 2 * 15 = 30 dollars. Adding the pants, the total is 30 + 25 = 55 "
    "dollars. The final answer is \\boxed{{55}}.\n\n"
    "Question:\n"
    "A 12 meter rope is cut into pieces that are each 3/4 of a meter long. How many "
    "pieces are there?\n\n"
    "Answer:\n"
    "The number of pieces is 12 divided by 3/4, which is 12 * 4 / 3 = 16. The final "
    "answer is \\boxed{{16}}.\n\n"
    "Question:\n"
    "A class has 40 students and 60% of them are girls. How many boys are in the "
    "class?\n\n"
    "Answer:\n"
    "There are 40 * 0.6 = 24 girls, so there are 40 - 24 = 16 boys. The final answer "
    "is \\boxed{{16}}.\n\n"
)

MATH_COT_BOXED_FEWSHOT_V1 = PromptTemplate(
    template_id="math_cot_boxed_fewshot_v1",
    required_domain="math",
    text=(
        "Solve each problem step by step and end with the final answer in the form "
        "\\boxed{{answer}}.\n\n" + _FEWSHOT_EXAMPLES + "Question:\n{question}\n\nAnswer:\n"
    ),
    metadata={"answer_format": "boxed", "num_shots": 4},
)
