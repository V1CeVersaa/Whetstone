import pytest
import torch

from whetstone.train.logprobs import masked_token_logprobs
from whetstone.train.losses import math_rl_loss, reinforce_loss, sampled_kl


def test_reinforce_loss_is_finite() -> None:
    torch.manual_seed(0)
    batch, seq_len, vocab = 4, 6, 12
    logits = torch.randn(batch, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    response_mask = torch.zeros(batch, seq_len, dtype=torch.long)
    response_mask[:, 2:] = 1
    advantages = torch.tensor([0.75, -0.25, -0.25, -0.25])

    losses = math_rl_loss(
        logits=logits,
        input_ids=input_ids,
        response_mask=response_mask,
        advantages=advantages,
    )
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["policy_loss"])
    assert float(losses["kl"]) == 0.0
    assert torch.allclose(losses["loss"], losses["policy_loss"])


def test_positive_advantage_prefers_higher_logprob() -> None:
    seq_logprobs = torch.tensor([-2.0, -2.0], requires_grad=True)
    advantages = torch.tensor([1.0, -1.0])
    loss = reinforce_loss(seq_logprobs, advantages)
    loss.backward()

    assert seq_logprobs.grad is not None
    # Gradient descent increases the logprob of the positive-advantage sample
    # (negative gradient) and decreases the negative-advantage one.
    assert seq_logprobs.grad[0] < 0
    assert seq_logprobs.grad[1] > 0

    # And a higher logprob for the positive sample means lower loss.
    better = reinforce_loss(torch.tensor([-1.0, -2.0]), advantages)
    assert float(better) < float(loss)


def test_zero_advantage_gives_zero_policy_loss() -> None:
    torch.manual_seed(1)
    seq_logprobs = torch.randn(6, requires_grad=True)
    loss = reinforce_loss(seq_logprobs, torch.zeros(6))
    assert float(loss) == 0.0
    loss.backward()
    assert seq_logprobs.grad is not None
    assert torch.all(seq_logprobs.grad == 0)


def test_kl_term_is_zero_when_policy_equals_reference() -> None:
    torch.manual_seed(2)
    batch, seq_len, vocab = 2, 5, 9
    logits = torch.randn(batch, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    response_mask = torch.zeros(batch, seq_len, dtype=torch.long)
    response_mask[:, 1:4] = 1

    token_logprobs = masked_token_logprobs(logits, input_ids, response_mask)
    assert float(sampled_kl(token_logprobs, token_logprobs.clone(), response_mask)) == 0.0

    losses = math_rl_loss(
        logits=logits,
        input_ids=input_ids,
        response_mask=response_mask,
        advantages=torch.zeros(batch),
        kl_beta=0.5,
        reference_token_logprobs=token_logprobs.detach().clone(),
    )
    assert float(losses["kl"]) == pytest.approx(0.0, abs=1e-6)


def test_kl_beta_requires_reference_logprobs() -> None:
    logits = torch.randn(1, 4, 5)
    input_ids = torch.randint(0, 5, (1, 4))
    response_mask = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="reference_token_logprobs"):
        math_rl_loss(
            logits=logits,
            input_ids=input_ids,
            response_mask=response_mask,
            advantages=torch.zeros(1),
            kl_beta=0.1,
        )
