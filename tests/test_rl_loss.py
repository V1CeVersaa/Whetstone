import torch

from whetstone.train.losses import math_rl_loss, reinforce_loss


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
