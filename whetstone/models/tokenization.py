from typing import Any

PREFERRED_DISTINCT_PAD_TOKENS = ("<|fim_pad|>", "<|pad|>", "<pad>", "<|image_pad|>")


def assign_pad_token(tokenizer: Any) -> Any:
    """Ensure a usable pad token, preferring one distinct from EOS.

    Pad positions are attention-masked and label-masked everywhere in
    Whetstone, so any existing special token is safe. Falls back to EOS when
    the vocab offers nothing better; downstream generation then guards the
    stop-string ambiguity by dropping to single-row batches.
    """
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        return tokenizer
    for candidate in PREFERRED_DISTINCT_PAD_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if (
            token_id is not None
            and token_id >= 0
            and token_id != tokenizer.eos_token_id
            and token_id != getattr(tokenizer, "unk_token_id", None)
        ):
            tokenizer.pad_token = candidate
            return tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def configure_tokenizer_for_generation(tokenizer: Any) -> Any:
    """Set left padding for batched decoder-only generation; return the tokenizer.

    Assigns a pad token distinct from EOS when the vocab has one (see
    :func:`assign_pad_token`). Left padding keeps generated tokens aligned
    across a batch.
    """
    tokenizer.padding_side = "left"
    return assign_pad_token(tokenizer)


def configure_tokenizer_for_training(tokenizer: Any) -> Any:
    """Set right padding for training collators; return the tokenizer.

    Assigns a pad token distinct from EOS when the vocab has one. Centralized
    here so padding policy is not scattered across scripts.
    """
    tokenizer.padding_side = "right"
    return assign_pad_token(tokenizer)
