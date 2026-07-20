"""Pad-token assignment policy: prefer a pad distinct from EOS.

pad == eos makes right-padding indistinguishable from a sampled EOS, which
forces single-row generation whenever stop_strings are active (observed on
Qwen3-Base + gsm8k_bench_fewshot: 200 batches of 1). These tests pin the
upgrade path and the honest fallback.
"""

from whetstone.models.tokenization import (
    assign_pad_token,
    configure_tokenizer_for_generation,
    configure_tokenizer_for_training,
)


class FakeHFTokenizer:
    """Minimal HF-like surface: setting pad_token updates pad_token_id."""

    def __init__(self, vocab: dict[str, int], *, eos_token: str, pad_token: str | None = None):
        self._vocab = vocab
        self.eos_token = eos_token
        self.eos_token_id = vocab[eos_token]
        self.unk_token_id = None
        self.padding_side = "right"
        self._pad_token: str | None = None
        self.pad_token_id: int | None = None
        if pad_token is not None:
            self.pad_token = pad_token

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        self._pad_token = value
        self.pad_token_id = self._vocab.get(value)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._vocab.get(token)


QWEN_LIKE_VOCAB = {"<|endoftext|>": 151643, "<|fim_pad|>": 151662}


def test_missing_pad_upgrades_to_distinct_special_token() -> None:
    tokenizer = configure_tokenizer_for_generation(
        FakeHFTokenizer(QWEN_LIKE_VOCAB, eos_token="<|endoftext|>")
    )
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<|fim_pad|>"
    assert tokenizer.pad_token_id != tokenizer.eos_token_id


def test_pad_equal_to_eos_is_upgraded() -> None:
    tokenizer = FakeHFTokenizer(
        QWEN_LIKE_VOCAB, eos_token="<|endoftext|>", pad_token="<|endoftext|>"
    )
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assign_pad_token(tokenizer)
    assert tokenizer.pad_token == "<|fim_pad|>"
    assert tokenizer.pad_token_id != tokenizer.eos_token_id


def test_existing_distinct_pad_is_untouched() -> None:
    vocab = {"</s>": 2, "<pad>": 0, "<|fim_pad|>": 9}
    tokenizer = FakeHFTokenizer(vocab, eos_token="</s>", pad_token="<pad>")
    assign_pad_token(tokenizer)
    assert tokenizer.pad_token == "<pad>"


def test_fallback_to_eos_when_vocab_has_no_candidate() -> None:
    tokenizer = configure_tokenizer_for_training(FakeHFTokenizer({"</s>": 2}, eos_token="</s>"))
    assert tokenizer.padding_side == "right"
    assert tokenizer.pad_token == "</s>"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
