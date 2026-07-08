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
    response_ids = response_ids[:max_response]
    max_prompt = max(1, max_seq_length - len(response_ids))
    prompt_ids = prompt_ids[-max_prompt:]
    return prompt_ids, response_ids


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
