import math

import torch

from whetstone.train.logprobs import (
    masked_token_logprobs,
    sequence_logprobs,
    token_logprobs_from_logits,
)


def test_token_logprobs_shape() -> None:
    batch, seq_len, vocab = 3, 7, 13
    logits = torch.randn(batch, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    token_logprobs = token_logprobs_from_logits(logits, input_ids)
    assert token_logprobs.shape == (batch, seq_len)
    # Position 0 has no prediction.
    assert (token_logprobs[:, 0] == 0).all()
    assert (token_logprobs[:, 1:] <= 0).all()


def test_uniform_logits_give_known_logprob() -> None:
    batch, seq_len, vocab = 2, 5, 8
    logits = torch.zeros(batch, seq_len, vocab)  # uniform distribution
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    token_logprobs = token_logprobs_from_logits(logits, input_ids)
    expected = -math.log(vocab)
    assert torch.allclose(token_logprobs[:, 1:], torch.full((batch, seq_len - 1), expected))


def test_sequence_logprob_masked_sum() -> None:
    torch.manual_seed(1)
    batch, seq_len, vocab = 2, 6, 10
    logits = torch.randn(batch, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    response_mask = torch.zeros(batch, seq_len, dtype=torch.long)
    response_mask[0, 3:] = 1
    response_mask[1, 2:5] = 1

    masked = masked_token_logprobs(logits, input_ids, response_mask)
    seq = sequence_logprobs(logits, input_ids, response_mask)
    assert torch.allclose(seq, masked.sum(dim=1))

    raw = token_logprobs_from_logits(logits, input_ids)
    expected = (raw * response_mask.float()).sum(dim=1)
    assert torch.allclose(seq, expected)


def test_padding_tokens_do_not_contribute() -> None:
    torch.manual_seed(2)
    batch, seq_len, vocab = 1, 8, 9
    logits = torch.randn(batch, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    response_mask = torch.zeros(batch, seq_len, dtype=torch.long)
    response_mask[0, 2:5] = 1  # positions 5..7 act as padding

    masked = masked_token_logprobs(logits, input_ids, response_mask)
    assert (masked[0, 5:] == 0).all()
    assert (masked[0, :2] == 0).all()

    # Replacing "padding" token ids must not change the masked logprobs.
    altered = input_ids.clone()
    altered[0, 6] = (altered[0, 6] + 1) % vocab
    assert torch.allclose(masked, masked_token_logprobs(logits, altered, response_mask))


def test_response_mask_controls_loss_region() -> None:
    torch.manual_seed(3)
    batch, seq_len, vocab = 1, 6, 7
    logits = torch.randn(batch, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    response_mask = torch.zeros(batch, seq_len, dtype=torch.long)
    response_mask[0, 4:] = 1

    # Changing a prompt token outside the mask leaves the masked accounting
    # untouched (for fixed logits): same masked values, same sequence sum.
    altered = input_ids.clone()
    altered[0, 1] = (altered[0, 1] + 3) % vocab
    original_seq = sequence_logprobs(logits, input_ids, response_mask)
    altered_seq = sequence_logprobs(logits, altered, response_mask)
    assert torch.allclose(original_seq, altered_seq)

    # Widening the mask changes which tokens are counted.
    wider = response_mask.clone()
    wider[0, 2:] = 1
    assert not torch.allclose(sequence_logprobs(logits, input_ids, wider), original_seq)
