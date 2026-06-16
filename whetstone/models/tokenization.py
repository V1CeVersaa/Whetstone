from typing import Any


def configure_tokenizer_for_generation(tokenizer: Any) -> Any:
    """Set left padding for batched decoder-only generation; return the tokenizer.

    Falls back to EOS as the pad token when the tokenizer has none. Left padding
    keeps generated tokens aligned across a batch.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def configure_tokenizer_for_training(tokenizer: Any) -> Any:
    """Set right padding for training collators; return the tokenizer.

    Falls back to EOS as the pad token when the tokenizer has none. Centralized
    here so padding policy is not scattered across scripts.
    """
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
