import torch
import torch.nn.functional as F


def token_logprobs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-position logprobs of the observed next tokens, aligned to ``input_ids``.

    Position ``t`` (for ``t >= 1``) holds ``log p(input_ids[:, t] | input_ids[:,
    :t])``, computed from ``logits[:, t-1]``. Position 0 has no prediction and
    is fixed to 0.

    Implementation note: this deliberately routes through
    ``F.cross_entropy(reduction="none")`` instead of materializing
    ``log_softmax`` over the vocabulary. With a 150k-vocab model and 2k-token
    sequences, an explicit float32 ``(batch, seq, vocab)`` log-softmax (plus
    its saved backward copy) costs several GiB and OOMs a 24GB GPU; the fused
    kernel computes the same values with float32 accumulation internally
    without retaining full-vocab float32 intermediates.

    Args:
        logits: ``(batch, seq_len, vocab)`` model outputs.
        input_ids: ``(batch, seq_len)`` token ids the logits were computed on.

    Returns:
        ``(batch, seq_len)`` float32 tensor of token logprobs.
    """
    vocab_size = logits.shape[-1]
    shifted_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    targets = input_ids[:, 1:].reshape(-1)
    negative_logprobs = F.cross_entropy(shifted_logits, targets, reduction="none")
    gathered = -negative_logprobs.reshape(input_ids.shape[0], -1).float()
    first = torch.zeros((input_ids.shape[0], 1), dtype=gathered.dtype, device=gathered.device)
    return torch.cat([first, gathered], dim=1)


def masked_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Token logprobs zeroed everywhere except response-token positions.

    ``response_mask`` marks the response positions in ``input_ids`` (the same
    convention as the collator), so prompt and padding tokens contribute
    exactly zero. This is the single loss-region definition shared by the SFT
    loss and the RL objective.
    """
    return token_logprobs_from_logits(logits, input_ids) * response_mask.to(dtype=torch.float32)


def sequence_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Summed response-token logprob per sequence: ``(batch,)`` float32.

    Equals the masked sum of :func:`masked_token_logprobs`, i.e.
    ``log pi(y | x)`` over exactly the generated/supervised response tokens.
    """
    return masked_token_logprobs(logits, input_ids, response_mask).sum(dim=1)
