import torch

from whetstone.train.logprobs import masked_token_logprobs


def sft_loss_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Response-token-only cross entropy, averaged over unmasked tokens.

    ``L = -(1 / sum(m)) * sum_t m_t * log p(y_t | x, y_<t)`` where ``m`` is the
    collator's response mask. Prompt and padding positions contribute nothing.
    Built on the same logprob utility Math-RL uses, so the two objectives share
    one loss-region definition.
    """
    token_logprobs = masked_token_logprobs(logits, input_ids, response_mask)
    num_response_tokens = response_mask.sum().clamp(min=1).to(dtype=torch.float32)
    return -token_logprobs.sum() / num_response_tokens


def reinforce_loss(
    seq_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """REINFORCE policy loss with externally supplied (group-baseline) advantages.

    ``L = -(1 / N) * sum_i stopgrad(A_i) * log pi(y_i | x_i)``. Advantages are
    detached inside, so gradients flow only through the sequence logprobs. A
    positive advantage pushes its sequence's logprob up; zero advantages
    contribute exactly zero loss and zero gradient.
    """
    return -(advantages.detach().to(seq_logprobs.dtype) * seq_logprobs).mean()


def sampled_kl(
    policy_token_logprobs: torch.Tensor,
    reference_token_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Sampled token-level KL estimate against a frozen reference.

    ``mean over response tokens of (log pi(y_t) - log pi_ref(y_t))``, evaluated
    only at the sampled tokens. This is an *approximation* of the true KL (it
    can even be negative on a batch), which is sufficient for Math-RL v0
    regularization and is logged as an estimate, not exact KL.
    """
    mask = response_mask.to(dtype=torch.float32)
    diff = (policy_token_logprobs - reference_token_logprobs) * mask
    return diff.sum() / mask.sum().clamp(min=1)


def math_rl_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_beta: float = 0.0,
    reference_token_logprobs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Full Math-RL v0 objective: REINFORCE term plus optional sampled KL.

    Returns a dict of scalars: ``policy_loss``, ``kl`` (0 when disabled), and
    ``loss`` (the term to backpropagate). ``reference_token_logprobs`` must be
    precomputed under ``no_grad`` and is required iff ``kl_beta > 0``.
    """
    policy_token_logprobs = masked_token_logprobs(logits, input_ids, response_mask)
    seq_logprobs = policy_token_logprobs.sum(dim=1)
    policy_loss = reinforce_loss(seq_logprobs, advantages)

    if kl_beta > 0.0:
        if reference_token_logprobs is None:
            msg = "kl_beta > 0 requires reference_token_logprobs"
            raise ValueError(msg)
        kl = sampled_kl(policy_token_logprobs, reference_token_logprobs, response_mask)
    else:
        kl = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)

    return {
        "policy_loss": policy_loss,
        "kl": kl,
        "loss": policy_loss + kl_beta * kl,
    }
