from statistics import fmean
from typing import Any

import torch

from whetstone.train.types import SFTExample

LABEL_IGNORE_INDEX = -100


def encode_prompt_response(
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
    *,
    max_seq_length: int = 2048,
    append_eos: bool = True,
) -> tuple[list[int], list[int]]:
    """Tokenize a prompt/response pair separately so the boundary is exact.

    The two segments are encoded without special tokens and concatenated by the
    collator, which is what makes the label mask boundary token-accurate. An
    EOS is appended to the response so the model learns to stop.

    Truncation keeps the response (the loss-bearing tokens) and trims the
    prompt from the left; a response longer than the budget is trimmed from
    its tail, always keeping at least one prompt and one response token.
    """
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    response_ids = list(tokenizer(response_text, add_special_tokens=False)["input_ids"])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos and eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids.append(eos_id)

    max_response = max(1, max_seq_length - 1)
    if len(response_ids) > max_response:
        response_ids = response_ids[:max_response]
        # Truncation must never silently drop the stop supervision: overwrite
        # the final kept token with EOS so every emitted target teaches the
        # model to stop. (Prefer dropping overlong examples entirely -- see
        # filter_overlong_sft_examples -- this is the safety net.)
        if append_eos and eos_id is not None:
            response_ids[-1] = eos_id
    max_prompt = max(1, max_seq_length - len(response_ids))
    prompt_ids = prompt_ids[-max_prompt:]
    return prompt_ids, response_ids


def filter_overlong_sft_examples(
    examples: list[SFTExample],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> tuple[list[SFTExample], int]:
    """Drop examples whose prompt+response+EOS exceed ``max_seq_length``.

    A truncated CoT target is a noisy training signal (it supervises an
    unfinished derivation), so the default SFT policy is to skip such examples
    and count them rather than train on them. Returns ``(kept, num_dropped)``.
    """
    kept: list[SFTExample] = []
    num_dropped = 0
    for example in examples:
        prompt_len = len(tokenizer(example.prompt_text, add_special_tokens=False)["input_ids"])
        response_ids = tokenizer(example.response_text, add_special_tokens=False)["input_ids"]
        response_len = len(response_ids)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_tokens = int(eos_id is not None and (not response_ids or response_ids[-1] != eos_id))
        if prompt_len + response_len + eos_tokens <= max_seq_length:
            kept.append(example)
        else:
            num_dropped += 1
    return kept, num_dropped


def audit_sft_tokenization(
    examples: list[SFTExample],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    """Audit the separate prompt/response tokenization policy on real inputs.

    The artifact deliberately reports rather than assumes equivalence with
    joint tokenization. It also records decode round-tripping and the length
    distribution that drives the configured overlong policy.
    """
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    total_lengths: list[int] = []
    separate_joint_mismatches: list[str] = []
    decode_mismatches: list[str] = []
    num_overlong = 0
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for example in examples:
        prompt_ids = list(tokenizer(example.prompt_text, add_special_tokens=False)["input_ids"])
        response_ids = list(tokenizer(example.response_text, add_special_tokens=False)["input_ids"])
        joint_ids = list(
            tokenizer(
                example.prompt_text + example.response_text,
                add_special_tokens=False,
            )["input_ids"]
        )
        separate_ids = prompt_ids + response_ids
        if separate_ids != joint_ids:
            separate_joint_mismatches.append(example.uid)
        if decode_token_ids(tokenizer, separate_ids) != example.prompt_text + example.response_text:
            decode_mismatches.append(example.uid)

        eos_tokens = int(eos_id is not None and (not response_ids or response_ids[-1] != eos_id))
        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(response_ids) + eos_tokens)
        total_length = len(prompt_ids) + len(response_ids) + eos_tokens
        total_lengths.append(total_length)
        if total_length > max_seq_length:
            num_overlong += 1

    count = len(examples)
    return {
        "tokenization_policy": "separate_prompt_response_v1",
        "num_examples": count,
        "max_seq_length": max_seq_length,
        "num_separate_joint_mismatches": len(separate_joint_mismatches),
        "separate_joint_mismatch_rate": len(separate_joint_mismatches) / max(1, count),
        "num_decode_mismatches": len(decode_mismatches),
        "decode_mismatch_rate": len(decode_mismatches) / max(1, count),
        "num_overlong": num_overlong,
        "overlong_rate": num_overlong / max(1, count),
        "prompt_tokens": length_summary(prompt_lengths),
        "response_tokens_including_eos": length_summary(response_lengths),
        "total_tokens": length_summary(total_lengths),
        "separate_joint_mismatch_uids_sample": separate_joint_mismatches[:20],
        "decode_mismatch_uids_sample": decode_mismatches[:20],
    }


def validate_sft_tokenization_audit(
    audit: dict[str, Any],
    *,
    max_decode_mismatch_rate: float = 0.0,
) -> None:
    """Reject a tokenization audit that violates the configured round-trip gate."""
    rate = float(audit.get("decode_mismatch_rate", 0.0))
    if rate > max_decode_mismatch_rate:
        sample = audit.get("decode_mismatch_uids_sample") or []
        msg = (
            f"SFT tokenization decode_mismatch_rate={rate:.6f} exceeds "
            f"preprocessing.max_decode_mismatch_rate={max_decode_mismatch_rate:.6f}; "
            f"sample_uids={sample}"
        )
        raise ValueError(msg)


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode without cleanup when the tokenizer exposes that option."""
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def length_summary(lengths: list[int]) -> dict[str, float | int]:
    """Compact deterministic length distribution for JSON artifacts."""
    if not lengths:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
        }
    ordered = sorted(lengths)

    def percentile(percent: int) -> int:
        index = round((len(ordered) - 1) * percent / 100)
        return ordered[index]

    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": fmean(ordered),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def collate_token_sequences(
    sequences: list[tuple[list[int], list[int]]],
    *,
    pad_token_id: int,
    label_ignore_index: int = LABEL_IGNORE_INDEX,
) -> dict[str, torch.Tensor]:
    """Right-pad ``(prompt_ids, response_ids)`` pairs into one training batch.

    This is the shared collation core: SFT feeds tokenized prompt/response
    pairs, Math-RL feeds prompt/sampled-completion token ids. The returned
    ``response_mask`` marks exactly the response token positions, and labels
    are ``input_ids`` with prompt and padding positions set to
    ``label_ignore_index`` -- the first response token is never masked.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``labels``, and
        ``response_mask``, all shaped ``(batch, max_len)``.
    """
    if not sequences:
        msg = "collate_token_sequences requires at least one sequence"
        raise ValueError(msg)
    lengths = [len(prompt) + len(response) for prompt, response in sequences]
    max_len = max(lengths)

    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    response_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)

    for row, (prompt_ids, response_ids) in enumerate(sequences):
        sequence = prompt_ids + response_ids
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1
        response_mask[row, len(prompt_ids) : len(sequence)] = 1

    labels = input_ids.clone()
    labels[response_mask == 0] = label_ignore_index
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_mask": response_mask,
    }


def collate_sft_examples(
    examples: list[SFTExample],
    tokenizer: Any,
    *,
    max_seq_length: int = 2048,
    label_ignore_index: int = LABEL_IGNORE_INDEX,
) -> dict[str, torch.Tensor]:
    """Tokenize and collate a batch of :class:`SFTExample` for one forward pass."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        msg = "Tokenizer has neither pad_token_id nor eos_token_id"
        raise ValueError(msg)
    sequences = [
        encode_prompt_response(
            tokenizer,
            example.prompt_text,
            example.response_text,
            max_seq_length=max_seq_length,
        )
        for example in examples
    ]
    return collate_token_sequences(
        sequences,
        pad_token_id=int(pad_token_id),
        label_ignore_index=label_ignore_index,
    )
