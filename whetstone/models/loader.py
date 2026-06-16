from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from whetstone.models.tokenization import configure_tokenizer_for_generation
from whetstone.utils.logging import get_logger

logger = get_logger(__name__)


def load_causal_lm(
    *,
    name_or_path: str,
    dtype: str = "bf16",
    device: str = "cuda",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer, configured for batched generation.

    The tokenizer is set to left padding (see
    :func:`~whetstone.models.tokenization.configure_tokenizer_for_generation`)
    and the model is moved to ``device`` and put in eval mode.

    Args:
        name_or_path: Hub id or local path, e.g. ``"Qwen/Qwen3-0.6B-Base"``.
        dtype: Weight dtype keyword (``bf16``/``fp16``/``fp32``/``auto``).
        device: Target device, or ``"auto"`` to leave placement to ``from_pretrained``.
        trust_remote_code: Whether to allow custom modeling code from the repo.

    Returns:
        A ``(model, tokenizer)`` tuple.
    """

    logger.info("Loading model %s (dtype=%s, device=%s)", name_or_path, dtype, device)

    torch_dtype = resolve_torch_dtype(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = configure_tokenizer_for_generation(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()

    logger.info("Model ready on %s", device)
    return model, tokenizer


def resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
    """Map a dtype keyword to a torch dtype (or the string ``"auto"``).

    Raises:
        ValueError: If ``dtype`` is not a recognized keyword.
    """
    normalized = dtype.lower()

    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    if normalized == "auto":
        return "auto"

    raise ValueError(f"Unsupported dtype: {dtype}")
