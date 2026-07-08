import torch

from whetstone.train.collators import (
    LABEL_IGNORE_INDEX,
    collate_sft_examples,
    collate_token_sequences,
    encode_prompt_response,
)
from whetstone.train.losses import sft_loss_from_logits
from whetstone.train.types import SFTExample


def sft_example(prompt: str, response: str) -> SFTExample:
    return SFTExample(uid="u", source="s", prompt_text=prompt, response_text=response)


def test_sft_response_masking(word_tokenizer) -> None:
    examples = [
        sft_example("Question: two plus three Answer:", "two plus three equals five"),
        sft_example("Question: short Answer:", "five"),
    ]
    batch = collate_sft_examples(examples, word_tokenizer, max_seq_length=64)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    response_mask = batch["response_mask"]
    attention_mask = batch["attention_mask"]

    for row, example in enumerate(examples):
        prompt_len = len(word_tokenizer(example.prompt_text)["input_ids"])
        response_len = len(word_tokenizer(example.response_text)["input_ids"]) + 1  # +EOS
        total = prompt_len + response_len

        # Prompt tokens are masked.
        assert (labels[row, :prompt_len] == LABEL_IGNORE_INDEX).all()
        # All response tokens (incl. the appended EOS) are unmasked...
        assert (labels[row, prompt_len:total] == input_ids[row, prompt_len:total]).all()
        assert (labels[row, prompt_len:total] != LABEL_IGNORE_INDEX).all()
        # ...and in particular the first response token is trainable.
        assert labels[row, prompt_len] != LABEL_IGNORE_INDEX
        assert response_mask[row, prompt_len] == 1
        # Padding tokens are masked.
        assert (labels[row, total:] == LABEL_IGNORE_INDEX).all()
        assert (attention_mask[row, total:] == 0).all()
        assert (attention_mask[row, :total] == 1).all()
        # EOS terminates the response.
        assert input_ids[row, total - 1] == word_tokenizer.eos_token_id


def test_sft_collator_padding(word_tokenizer) -> None:
    examples = [
        sft_example("a much longer prompt with many words here", "long response text"),
        sft_example("tiny", "ok"),
    ]
    batch = collate_sft_examples(examples, word_tokenizer, max_seq_length=64)
    lengths = [
        len(word_tokenizer(ex.prompt_text)["input_ids"])
        + len(word_tokenizer(ex.response_text)["input_ids"])
        + 1
        for ex in examples
    ]
    assert batch["input_ids"].shape == (2, max(lengths))
    # Short row is right-padded with pad_token_id, masked everywhere.
    short_row, short_len = 1, lengths[1]
    assert (batch["input_ids"][short_row, short_len:] == word_tokenizer.pad_token_id).all()
    assert (batch["response_mask"][short_row, short_len:] == 0).all()
    assert int(batch["attention_mask"].sum()) == sum(lengths)


def test_truncation_keeps_response_and_at_least_one_prompt_token(word_tokenizer) -> None:
    prompt = " ".join(f"p{i}" for i in range(20))
    response = " ".join(f"r{i}" for i in range(10))
    prompt_ids, response_ids = encode_prompt_response(
        word_tokenizer, prompt, response, max_seq_length=12
    )
    assert len(prompt_ids) + len(response_ids) <= 12
    assert len(prompt_ids) >= 1
    assert len(response_ids) == 11  # response (10) + EOS fits the 12-token budget
    # Prompt is trimmed from the left: the kept token is the final prompt token.
    assert prompt_ids == word_tokenizer(prompt)["input_ids"][-1:]


def test_sft_loss_ignores_prompt_tokens() -> None:
    torch.manual_seed(0)
    vocab = 11
    sequences = [([3, 4, 5], [6, 7]), ([3], [8, 9, 10])]
    batch = collate_token_sequences(sequences, pad_token_id=0)
    logits = torch.randn(2, batch["input_ids"].shape[1], vocab, requires_grad=True)

    loss = sft_loss_from_logits(logits, batch["input_ids"], batch["response_mask"])
    loss.backward()
    assert logits.grad is not None

    # The logits at position t-1 predict token t; gradients may only flow where
    # the *predicted* position is a response token.
    for row in range(2):
        for position in range(batch["input_ids"].shape[1] - 1):
            grad_norm = float(logits.grad[row, position].abs().sum())
            if batch["response_mask"][row, position + 1] == 1:
                assert grad_norm > 0
            else:
                assert grad_norm == 0

    # Changing prompt token ids does not change the loss for fixed logits.
    altered_ids = batch["input_ids"].clone()
    altered_ids[0, 1] = 9  # a masked prompt target position
    unchanged = sft_loss_from_logits(
        logits.detach(), altered_ids, batch["response_mask"]
    )
    original = sft_loss_from_logits(
        logits.detach(), batch["input_ids"], batch["response_mask"]
    )
    assert torch.allclose(unchanged, original)
