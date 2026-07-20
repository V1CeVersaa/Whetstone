"""Training objectives, exposed in two forms.

The ``*_terms`` functions return **unnormalized sums plus their denominators**.
Sums are additive across micro-batches while means are not, so gradient
accumulation (SFT) and the chunked RL update accumulate these terms and
normalize once by the global denominators -- which reproduces the full-batch
objective exactly, independent of chunking. The mean-form functions
(``sft_loss_from_logits``, ``math_rl_loss``) are thin wrappers over the same
terms for single-batch use and tests; they can never drift from what the
training loops optimize, because both derive from one definition.
"""

from dataclasses import dataclass

import torch

from whetstone.train.logprobs import masked_token_logprobs


@dataclass(frozen=True)
class MathRLLossTerms:
    """Unnormalized Math-RL loss pieces plus their denominators.

    Additive across chunks: sum the sums, sum the counts, normalize once.
    """

    policy_loss_sum: torch.Tensor
    num_sequences: int
    num_response_tokens: int


def sft_loss_terms(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Unnormalized SFT pieces: (NLL sum over response tokens, token count).

    ``sum(nll_sums) / sum(counts)`` over micro-batches equals the full-batch
    token-mean cross entropy; averaging per-micro-batch means does not (it
    up-weights short-response micro-batches per token).
    """
    token_logprobs = masked_token_logprobs(logits, input_ids, response_mask)
    return -token_logprobs.sum(), int(response_mask.sum())


def sft_loss_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Response-token-only cross entropy, averaged over unmasked tokens.

    ``L = -(1 / sum(m)) * sum_t m_t * log p(y_t | x, y_<t)`` where ``m`` is the
    collator's response mask. Prompt and padding positions contribute nothing.
    """
    nll_sum, num_response_tokens = sft_loss_terms(logits, input_ids, response_mask)
    return nll_sum / max(1, num_response_tokens)


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


def math_rl_loss_terms(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> MathRLLossTerms:
    """Unnormalized Math-RL pieces for exact cross-chunk accumulation.

    ``policy_loss_sum`` is ``-sum_i A_i * log pi(y_i|x_i)`` over sequences.
    A chunked update backprops ``policy_loss_sum / total_sequences`` per chunk;
    the accumulated gradient equals the full-batch one by construction.
    """
    policy_token_logprobs = masked_token_logprobs(logits, input_ids, response_mask)
    seq_logprobs = policy_token_logprobs.sum(dim=1)
    policy_loss_sum = -(advantages.detach().to(seq_logprobs.dtype) * seq_logprobs).sum()

    return MathRLLossTerms(
        policy_loss_sum=policy_loss_sum,
        num_sequences=int(seq_logprobs.shape[0]),
        num_response_tokens=int(response_mask.sum()),
    )


def math_rl_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Full-batch Math-RL v0 objective: group-baseline REINFORCE.

    Thin normalization of :func:`math_rl_loss_terms`; correct when the whole
    batch goes through one forward pass. The chunked training loop uses the
    terms directly. KL is intentionally absent from Phase 1; the future
    implementation enters through detached reward shaping, not this loss.
    """
    terms = math_rl_loss_terms(
        logits=logits,
        input_ids=input_ids,
        response_mask=response_mask,
        advantages=advantages,
    )
    policy_loss = terms.policy_loss_sum / max(1, terms.num_sequences)
    return {
        "policy_loss": policy_loss,
        "loss": policy_loss,
    }
