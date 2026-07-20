from whetstone.core.types import WhetstoneExample

TINY_MATH_ROWS = [
    {
        "uid": "tiny_math:validation:addition",
        "prompt": "Mira has 2 marbles and buys 3 more. How many marbles does she have?",
        "solution": "Mira starts with 2 marbles and buys 3 more, so 2 + 3 = 5.",
        "answer": "5",
    },
    {
        "uid": "tiny_math:validation:fraction",
        "prompt": "What is one half plus one quarter?",
        "solution": "One half is 2/4, and 2/4 + 1/4 = 3/4.",
        "answer": "3/4",
    },
    {
        "uid": "tiny_math:validation:decimal",
        "prompt": "A rope is 7.5 meters long and is cut in half. How long is each piece?",
        "solution": "Each piece is 7.5 / 2 = 3.75 meters.",
        "answer": "3.75",
    },
]

TINY_CODE_ROWS = [
    {
        "uid": "tiny_code:validation:echo",
        "prompt": "Read all input from standard input and print it unchanged.",
        "solution": "import sys\nprint(sys.stdin.read().rstrip())\n",
        "tests": {
            "public": [
                {"input": "hello\n", "output": "hello\n"},
                {"input": "whetstone\n", "output": "whetstone\n"},
            ],
            "hidden": [{"input": "42\n", "output": "42\n"}],
        },
    },
    {
        "uid": "tiny_code:validation:sum_two",
        "prompt": "Read two integers separated by whitespace and print their sum.",
        "solution": "import sys\nnums = list(map(int, sys.stdin.read().split()))\nprint(sum(nums))\n",
        "tests": {
            "public": [
                {"input": "2 3\n", "output": "5\n"},
                {"input": "-4 10\n", "output": "6\n"},
            ],
            "hidden": [{"input": "100 250\n", "output": "350\n"}],
        },
    },
]


class TinyMathAdapter:
    """Built-in math fixture dataset for fully offline artifact smoke tests."""

    name = "tiny_math"
    domain = "math"

    def load(self, split: str, limit: int | None = None) -> list[WhetstoneExample]:
        rows = TINY_MATH_ROWS[:limit] if limit is not None else TINY_MATH_ROWS
        return [
            WhetstoneExample(
                uid=replace_split(str(row["uid"]), split),
                domain="math",
                source="tiny_math",
                split=split,
                prompt_raw=str(row["prompt"]),
                reference_solution=str(row["solution"]),
                final_answer=str(row["answer"]),
                metadata={"dataset": "tiny_math", "row_index": index, "fixture": True},
            )
            for index, row in enumerate(rows)
        ]


class TinyCodeAdapter:
    """Built-in code fixture dataset for fully offline artifact smoke tests."""

    name = "tiny_code"
    domain = "code"

    def load(self, split: str, limit: int | None = None) -> list[WhetstoneExample]:
        rows = TINY_CODE_ROWS[:limit] if limit is not None else TINY_CODE_ROWS
        return [
            WhetstoneExample(
                uid=replace_split(str(row["uid"]), split),
                domain="code",
                source="tiny_code",
                split=split,
                prompt_raw=str(row["prompt"]),
                reference_solution=str(row["solution"]),
                tests=dict(row["tests"]),
                metadata={
                    "dataset": "tiny_code",
                    "row_index": index,
                    "fixture": True,
                    "num_public_tests": len(row["tests"]["public"]),
                    "num_hidden_tests": len(row["tests"]["hidden"]),
                },
            )
            for index, row in enumerate(rows)
        ]


def replace_split(uid: str, split: str) -> str:
    parts = uid.split(":")
    if len(parts) >= 3:
        parts[1] = split
        return ":".join(parts)
    return uid
