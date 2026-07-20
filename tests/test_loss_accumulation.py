"""Chunked accumulation must reproduce the full-batch objective exactly.

These tests pin the *_terms contract: accumulating unnormalized sums across
row-chunks and normalizing once by the global denominators gives the same
loss value and the same gradients as a single full-batch computation. Rows
deliberately have different response lengths, which is exactly the case where
equal-weighting per-chunk means diverges from the true objective.
"""

import torch

from whetstone.train.collators import collate_token_sequences
from whetstone.train.losses import (
    math_rl_loss,
    math_rl_loss_terms,
    sft_loss_from_logits,
    sft_loss_terms,
)

VOCAB = 13

SEQUENCES = [
    ([1, 2, 3], [4, 5]),
    ([1], [6, 7, 8, 9]),
    ([2, 2], [3]),
    ([5, 6, 7, 8], [9, 10, 11]),
]


def make_batch():
    torch.manual_seed(0)
    batch = collate_token_sequences(SEQUENCES, pad_token_id=0)
    logits = torch.randn(len(SEQUENCES), batch["input_ids"].shape[1], VOCAB)
    return batch, logits


def test_sft_accumulated_terms_match_full_batch() -> None:
    batch, base_logits = make_batch()

    full_logits = base_logits.clone().requires_grad_(True)
    full_loss = sft_loss_from_logits(full_logits, batch["input_ids"], batch["response_mask"])
    full_loss.backward()

    chunk_logits = base_logits.clone().requires_grad_(True)
    total_tokens = int(batch["response_mask"].sum())
    accumulated = 0.0
    for rows in (slice(0, 1), slice(1, 3), slice(3, 4)):  # uneven chunks on purpose
        nll_sum, _ = sft_loss_terms(
            chunk_logits[rows], batch["input_ids"][rows], batch["response_mask"][rows]
        )
        (nll_sum / total_tokens).backward()
        accumulated += float(nll_sum.detach())

    assert abs(accumulated / total_tokens - float(full_loss.detach())) < 1e-6
    assert full_logits.grad is not None and chunk_logits.grad is not None
    assert torch.allclose(full_logits.grad, chunk_logits.grad, atol=1e-6)


def test_math_rl_accumulated_terms_match_full_batch() -> None:
    batch, base_logits = make_batch()
    advantages = torch.tensor([0.75, -0.25, -0.25, 0.5])

    full_logits = base_logits.clone().requires_grad_(True)
    full = math_rl_loss(
        logits=full_logits,
        input_ids=batch["input_ids"],
        response_mask=batch["response_mask"],
        advantages=advantages,
    )
    full["loss"].backward()

    chunk_logits = base_logits.clone().requires_grad_(True)
    total_sequences = len(SEQUENCES)
    policy_sum = 0.0
    for rows in (slice(0, 2), slice(2, 4)):
        terms = math_rl_loss_terms(
            logits=chunk_logits[rows],
            input_ids=batch["input_ids"][rows],
            response_mask=batch["response_mask"][rows],
            advantages=advantages[rows],
        )
        chunk_loss = terms.policy_loss_sum / total_sequences
        chunk_loss.backward()
        policy_sum += float(terms.policy_loss_sum.detach())

    assert abs(policy_sum / total_sequences - float(full["policy_loss"].detach())) < 1e-6
    assert full_logits.grad is not None and chunk_logits.grad is not None
    assert torch.allclose(full_logits.grad, chunk_logits.grad, atol=1e-6)


def test_terms_report_global_denominators() -> None:
    batch, base_logits = make_batch()
    terms = math_rl_loss_terms(
        logits=base_logits,
        input_ids=batch["input_ids"],
        response_mask=batch["response_mask"],
        advantages=torch.zeros(len(SEQUENCES)),
    )
    assert terms.num_sequences == len(SEQUENCES)
    assert terms.num_response_tokens == int(batch["response_mask"].sum())
