from whetstone.core.types import WhetstoneExample
from whetstone.data import get_dataset_domain
from whetstone.data.base import load_hf_rows
from whetstone.data.gsm8k import GSM8KAdapter
from whetstone.data.openr1_math import OpenR1MathAdapter
from whetstone.data.taco_cobalt import TacoCobaltAdapter
from whetstone.data.tiny import TinyCodeAdapter, TinyMathAdapter


def test_gsm8k_emits_whetstone_example() -> None:
    adapter = GSM8KAdapter(rows=[{"question": "1+1?", "answer": "1+1=2\n#### 2"}])
    example = adapter.load(split="train", limit=1)[0]
    assert isinstance(example, WhetstoneExample)
    assert example.domain == "math"
    assert example.final_answer == "2"


def test_openr1_emits_whetstone_example() -> None:
    adapter = OpenR1MathAdapter(
        rows=[
            {
                "uuid": "abc",
                "problem": "Compute 2+2.",
                "solution": "It is 4.",
                "answer": "4",
                "problem_type": "arithmetic",
            }
        ]
    )
    example = adapter.load(split="train", limit=1)[0]
    assert example.uid == "openr1_math:train:abc"
    assert example.domain == "math"
    assert example.metadata["problem_type"] == "arithmetic"


def test_taco_cobalt_emits_code_example_with_tests() -> None:
    adapter = TacoCobaltAdapter(
        rows=[
            {
                "id": "task1",
                "question": "Echo input.",
                "public_test_cases": '{"inputs": ["a\\n"], "outputs": ["a\\n"]}',
                "hidden_test_cases": [{"input": "b\n", "output": "b\n"}],
                "difficulty": "easy",
            }
        ]
    )
    example = adapter.load(split="validation", limit=1)[0]
    assert example.uid == "taco_cobalt:validation:task1"
    assert example.domain == "code"
    assert example.tests is not None
    assert example.tests["public"] == [{"input": "a\n", "output": "a\n"}]
    assert example.tests["hidden"] == [{"input": "b\n", "output": "b\n"}]


def test_tiny_math_is_offline_fixture_dataset() -> None:
    example = TinyMathAdapter().load(split="validation", limit=1)[0]
    assert example.uid == "tiny_math:validation:addition"
    assert example.domain == "math"
    assert example.final_answer == "5"
    assert example.metadata["fixture"] is True


def test_tiny_code_is_offline_fixture_dataset() -> None:
    example = TinyCodeAdapter().load(split="validation", limit=1)[0]
    assert example.uid == "tiny_code:validation:echo"
    assert example.domain == "code"
    assert example.tests is not None
    assert len(example.tests["public"]) == 2
    assert example.metadata["fixture"] is True


def test_dataset_domains_are_available_without_loading_rows() -> None:
    assert get_dataset_domain("gsm8k") == "math"
    assert get_dataset_domain("openr1-math") == "math"
    assert get_dataset_domain("taco_cobalt") == "code"


def test_load_hf_rows_pushes_limit_into_non_streaming_split(monkeypatch) -> None:
    calls = {}

    def fake_load_dataset(dataset_name, *, name, split, data_files, streaming):
        calls.update(
            {
                "dataset_name": dataset_name,
                "name": name,
                "split": split,
                "data_files": data_files,
                "streaming": streaming,
            }
        )
        return [{"x": 1}, {"x": 2}]

    monkeypatch.setattr("whetstone.data.base.load_dataset", fake_load_dataset)

    rows = load_hf_rows("repo/dataset", "train", 2)

    assert rows == [{"x": 1}, {"x": 2}]
    assert calls["dataset_name"] == "repo/dataset"
    assert calls["split"] == "train[:2]"
    assert calls["streaming"] is False


def test_load_hf_rows_keeps_streaming_split_and_takes_rows(monkeypatch) -> None:
    calls = {}

    def fake_load_dataset(dataset_name, *, name, split, data_files, streaming):
        calls.update({"dataset_name": dataset_name, "split": split, "streaming": streaming})
        return iter([{"x": 1}, {"x": 2}, {"x": 3}])

    monkeypatch.setattr("whetstone.data.base.load_dataset", fake_load_dataset)

    rows = load_hf_rows("repo/dataset", "train", 2, streaming=True)

    assert rows == [{"x": 1}, {"x": 2}]
    assert calls["split"] == "train"
    assert calls["streaming"] is True
