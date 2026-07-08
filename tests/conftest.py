"""Shared offline test doubles for the training-layer tests.

``WordTokenizer`` is a deterministic whitespace tokenizer implementing exactly
the surface the train collators and the fake rollout path use. ``tiny_causal_lm``
builds a randomly initialized 2-layer GPT-2 from config (no download), small
enough for CPU overfit/integration tests.
"""

import pytest


class WordTokenizer:
    """Whitespace tokenizer double: pad=0, eos=1, words assigned ids from 2 up."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"
        self._vocab: dict[str, int] = {}

    def _word_id(self, word: str) -> int:
        return self._vocab.setdefault(word, len(self._vocab) + 2)

    def _encode(self, text: str) -> list[int]:
        return [self._word_id(word) for word in text.split()]

    def __call__(self, text, add_special_tokens: bool = False, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [self._encode(item) for item in text]}
        return {"input_ids": self._encode(text)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        reverse = {token_id: word for word, token_id in self._vocab.items()}
        words = [reverse[i] for i in ids if i in reverse]
        return " ".join(words)


@pytest.fixture
def word_tokenizer() -> WordTokenizer:
    return WordTokenizer()


@pytest.fixture
def tiny_causal_lm():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=512,
        n_positions=256,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
        # Deterministic forward passes: the RL tests compare policy and
        # reference logits computed from identical weights.
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(config)
