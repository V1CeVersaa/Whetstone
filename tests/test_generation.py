from types import SimpleNamespace

import pytest
import torch

from whetstone.core.types import RenderedPrompt
from whetstone.models.generation import (
    compact_pad_runs,
    generate_completions,
    matched_stop_string,
    model_generate_kwargs,
)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, texts, *, padding: bool, return_tensors: str):
        assert padding is True
        assert return_tensors == "pt"

        sequences = [[11, 12], [21, 22, 23]][: len(texts)]
        width = max(len(sequence) for sequence in sequences)
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            pad_count = width - len(sequence)
            input_ids.append([0] * pad_count + sequence)
            attention_mask.append([0] * pad_count + [1] * len(sequence))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def decode(self, ids, *, skip_special_tokens: bool):
        values = [int(value) for value in ids.tolist()]
        if skip_special_tokens:
            values = [value for value in values if value not in {0, self.eos_token_id}]
        vocab = {
            11: "short",
            12: "prompt",
            21: "long",
            22: "prompt",
            23: "tokens",
            31: "first",
            32: "answer",
            41: "second",
            42: "answer",
            43: "done",
        }
        return " ".join(vocab.get(value, f"<{value}>") for value in values)


class FakeModel:
    def generate(self, *, input_ids, attention_mask, **kwargs):
        completions = torch.tensor(
            [
                [31, 32, 99, 0],
                [41, 42, 43, 99],
            ]
        )
        return torch.cat([input_ids, completions[: input_ids.shape[0]]], dim=1)


class FakeMultiSequenceModel:
    def generate(self, *, input_ids, attention_mask, **kwargs):
        completion = torch.tensor([[31, 99]])
        return torch.cat([input_ids, completion], dim=1).repeat(2, 1)


def test_generate_completions_slices_after_left_padded_prompt_width() -> None:
    prompts = [
        RenderedPrompt(uid="u1", template_id="t", text="short prompt"),
        RenderedPrompt(uid="u2", template_id="t", text="long prompt tokens"),
    ]

    completions = generate_completions(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        prompts=prompts,
        generation_config={"max_new_tokens": 4},
        model_name_or_path="fake",
        device=None,
    )

    assert [completion.uid for completion in completions] == ["u1", "u2"]
    assert completions[0].completion == "first answer"
    assert completions[0].num_prompt_tokens == 2
    assert completions[0].num_completion_tokens == 3
    assert completions[0].finish_reason == "stop"
    assert completions[1].completion == "second answer done"
    assert completions[1].num_prompt_tokens == 3
    assert completions[1].num_completion_tokens == 4


def test_generate_completions_rejects_multiple_return_sequences() -> None:
    prompts = [RenderedPrompt(uid="u1", template_id="t", text="short prompt")]

    with pytest.raises(ValueError, match="num_return_sequences"):
        generate_completions(
            model=FakeMultiSequenceModel(),
            tokenizer=FakeTokenizer(),
            prompts=prompts,
            generation_config={"num_return_sequences": 2},
            model_name_or_path="fake",
            device=None,
        )


class FakeStopStringModel:
    """Simulates a row ended early by stop_strings: right-padded, no EOS."""

    def generate(self, *, input_ids, attention_mask, **kwargs):
        completion = torch.tensor([[31, 32, 0, 0]])  # pad=0 follows real tokens, no eos
        return torch.cat([input_ids, completion[: input_ids.shape[0]]], dim=1)


def test_stop_string_padding_is_not_counted_as_completion() -> None:
    prompts = [RenderedPrompt(uid="u1", template_id="t", text="short prompt")]

    completions = generate_completions(
        model=FakeStopStringModel(),
        tokenizer=FakeTokenizer(),
        prompts=prompts,
        generation_config={"max_new_tokens": 4, "stop_strings": ["\nQuestion:"]},
        model_name_or_path="fake",
        device=None,
    )

    # Trailing pads (pad != eos) are cut, not counted, and the finish reason
    # reflects the stop-string stop instead of pretending the row ran to length.
    assert completions[0].num_completion_tokens == 2
    assert completions[0].completion == "first answer"
    assert completions[0].finish_reason == "stop_string"


def test_generate_completions_chunks_by_batch_size() -> None:
    class CountingModel(FakeModel):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            return super().generate(**kwargs)

    prompts = [
        RenderedPrompt(uid="u1", template_id="t", text="short prompt"),
        RenderedPrompt(uid="u2", template_id="t", text="long prompt tokens"),
    ]
    model = CountingModel()
    completions = generate_completions(
        model=model,
        tokenizer=FakeTokenizer(),
        prompts=prompts,
        generation_config={"max_new_tokens": 4, "batch_size": 1},
        model_name_or_path="fake",
        device=None,
    )

    assert model.calls == 2
    assert [completion.uid for completion in completions] == ["u1", "u2"]


def test_generation_temporarily_restores_cache_and_model_mode() -> None:
    class StatefulModel(FakeModel):
        def __init__(self) -> None:
            self.training = True
            self.config = SimpleNamespace(use_cache=False)

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

        def generate(self, **kwargs):
            assert self.training is False
            assert self.config.use_cache is True
            return super().generate(**kwargs)

    model = StatefulModel()
    generate_completions(
        model=model,
        tokenizer=FakeTokenizer(),
        prompts=[RenderedPrompt(uid="u1", template_id="t", text="short prompt")],
        generation_config={"max_new_tokens": 4},
        model_name_or_path="fake",
    )

    assert model.training is True
    assert model.config.use_cache is False


def test_stop_strings_force_single_row_batches_when_pad_equals_eos() -> None:
    class SamePadTokenizer(FakeTokenizer):
        pad_token_id = FakeTokenizer.eos_token_id

    class CountingUnpaddedModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *, input_ids, attention_mask, **kwargs):
            self.calls += 1
            completion = torch.tensor([[31, 32]])
            return torch.cat([input_ids, completion], dim=1)

    model = CountingUnpaddedModel()
    generate_completions(
        model=model,
        tokenizer=SamePadTokenizer(),
        prompts=[
            RenderedPrompt(uid="u1", template_id="t", text="short prompt"),
            RenderedPrompt(uid="u2", template_id="t", text="long prompt tokens"),
        ],
        generation_config={
            "max_new_tokens": 4,
            "batch_size": 2,
            "stop_strings": ["\nQuestion:"],
        },
        model_name_or_path="fake",
    )

    assert model.calls == 2


def test_decoded_stop_string_is_detected_without_padding() -> None:
    assert matched_stop_string("reasoning\nQuestion:", {"stop_strings": ["\nQuestion:"]})
    assert not matched_stop_string("reasoning", {"stop_strings": ["\nQuestion:"]})


def test_pad_runs_are_compacted_in_provenance_text() -> None:
    pad = "<|fim_pad|>"
    assert compact_pad_runs(pad * 46 + "text" + pad * 2, pad) == f"{pad}*46text{pad}*2"
    # Single occurrences stay literal; unrelated text and pads are untouched.
    assert compact_pad_runs(pad + "text", pad) == pad + "text"
    assert compact_pad_runs("text", pad) == "text"
    assert compact_pad_runs(pad * 9, None) == pad * 9


def test_model_generate_kwargs_drops_sampling_only_fields_for_greedy_generation() -> None:
    kwargs = model_generate_kwargs(
        {
            "backend": "mock",
            "batch_size": 32,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    )
    assert kwargs == {"do_sample": False}
