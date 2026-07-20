from types import SimpleNamespace

import pytest
import torch

from whetstone.algorithms.math_rl import build_update_sequences
from whetstone.rollout.math_rollout import _generate_real


class RolloutTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, texts, *, padding: bool, return_tensors: str):
        assert padding is True
        assert return_tensors == "pt"
        rows = [[10 + index] for index, _ in enumerate(texts)]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.ones((len(rows), 1), dtype=torch.long),
        }

    def decode(self, ids, *, skip_special_tokens: bool) -> str:
        values = [int(value) for value in ids]
        values = [value for value in values if value not in {self.eos_token_id, self.pad_token_id}]
        return " ".join(str(value) for value in values)


class GroupedRolloutModel:
    def __init__(self) -> None:
        self.calls = 0
        self.training = True
        self.config = SimpleNamespace(use_cache=False)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, *, input_ids, attention_mask, num_return_sequences: int, **kwargs):
        assert self.training is False
        assert self.config.use_cache is True
        self.calls += 1
        repeated = input_ids.repeat_interleave(num_return_sequences, dim=0)
        completions = torch.tensor([[31 + member, 99] for member in range(num_return_sequences)])
        return torch.cat([repeated, completions], dim=1)


def test_grouped_rollout_generation_honors_prompt_batch_size_and_restores_model() -> None:
    model = GroupedRolloutModel()
    prompt_ids, completion_ids, completion_texts = _generate_real(
        model=model,
        tokenizer=RolloutTokenizer(),
        prompt_texts=["p0", "p1", "p2"],
        group_size=2,
        generation_config={
            "batch_size": 1,
            "max_new_tokens": 2,
            "do_sample": True,
            "temperature": 0.7,
        },
        device=None,
    )

    assert model.calls == 3
    assert len(prompt_ids) == 3
    assert len(completion_ids) == 6
    assert completion_texts == ["31", "32"] * 3
    assert model.training is True
    assert model.config.use_cache is False


def test_update_sequences_never_change_rollout_context_or_completion() -> None:
    with pytest.raises(ValueError, match="never truncated"):
        build_update_sequences([[10, 11]], [[20, 21, 22, 23]], max_seq_length=4)

    with pytest.raises(ValueError, match="rollout context"):
        build_update_sequences([[10, 11, 12]], [[20, 21]], max_seq_length=4)

    assert build_update_sequences(
        [[10, 11, 12]],
        [[20, 21]],
        max_seq_length=5,
    ) == [([10, 11, 12], [20, 21])]
