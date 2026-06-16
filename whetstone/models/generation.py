from collections.abc import Mapping
from typing import Any

import torch

from whetstone.core.types import ModelCompletion, RenderedPrompt, WhetstoneExample
from whetstone.utils.logging import get_logger

logger = get_logger(__name__)


def generate_completions(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[RenderedPrompt],
    generation_config: Mapping[str, Any],
    model_name_or_path: str,
    device: str | None = None,
) -> list[ModelCompletion]:
    """Generate one completion per prompt and return structured results.

    Prompts are batched with left padding, so generated tokens for every
    sequence begin at the same padded prompt width. Token counts exclude padding
    and ``finish_reason`` reflects whether an EOS token was produced.

    Args:
        model: A loaded causal LM.
        tokenizer: Its tokenizer, configured for left-padded generation.
        prompts: Rendered prompts to complete.
        generation_config: Decoding kwargs passed to ``model.generate`` (a
            ``backend`` key, if present, is ignored).
        model_name_or_path: Recorded in each completion's metadata.
        device: Device to move inputs to; ``None``/``"auto"`` leaves them as-is.

    Returns:
        One :class:`ModelCompletion` per prompt, in order.
    """
    logger.info(
        f"Generating {len(prompts)} completions (max_new_tokens={generation_config.get('max_new_tokens')})"
    )

    prompt_texts = [prompt.text for prompt in prompts]
    encoded = tokenizer(prompt_texts, padding=True, return_tensors="pt")
    if device is not None and device != "auto":
        encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

    gen_kwargs = model_generate_kwargs(generation_config)
    with torch.no_grad():
        generated = model.generate(**encoded, **gen_kwargs)
    if len(generated) != len(prompts):
        msg = (
            f"Expected one generated sequence per prompt, got {len(generated)} "
            f"for {len(prompts)} prompts (num_return_sequences > 1 is unsupported)"
        )
        raise ValueError(msg)
    completions: list[ModelCompletion] = []

    eos_id = tokenizer.eos_token_id
    for index, prompt in enumerate(prompts):
        output_ids = generated[index]
        full_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        completion_ids = output_ids[prompt_width:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        # Count up to and including the first EOS; everything after it is right
        # padding. Keying off EOS position (not pad_id) stays correct even when
        # pad_token_id == eos_token_id, where filtering pads would also drop the
        # real EOS and undercount by one.

        stopped = False
        num_completion_tokens = int(completion_ids.numel())
        if eos_id is not None:
            eos_hits = (completion_ids == eos_id).nonzero(as_tuple=True)[0]
            if eos_hits.numel() > 0:
                stopped = True
                num_completion_tokens = int(eos_hits[0].item()) + 1

        completions.append(
            ModelCompletion(
                uid=prompt.uid,
                completion=completion_text,
                full_text=full_text,
                num_prompt_tokens=int(prompt_lengths[index]),
                num_completion_tokens=num_completion_tokens,
                finish_reason="stop" if stopped else "length",
                generation_metadata={
                    "model_name_or_path": model_name_or_path,
                    "generation_config": dict(generation_config),
                },
            )
        )
    return completions


def model_generate_kwargs(generation_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return kwargs for ``model.generate`` while preserving artifact config elsewhere."""
    kwargs = dict(generation_config)
    kwargs.pop("backend", None)
    if kwargs.get("do_sample") is False:
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
    return kwargs


def generate_mock_completions(
    *,
    examples: list[WhetstoneExample],
    prompts: list[RenderedPrompt],
    mode: str = "reference",
) -> list[ModelCompletion]:
    """Produce deterministic stand-in completions without loading a model.

    Used to debug the artifact pipeline end-to-end before downloading weights.
    ``mode="reference"`` echoes the gold answer/solution; ``mode="empty"`` returns
    empty completions to exercise failure paths.
    """
    completions: list[ModelCompletion] = []
    for example, prompt in zip(examples, prompts, strict=True):
        completion = mock_completion_text(example, mode=mode)
        completions.append(
            ModelCompletion(
                uid=prompt.uid,
                completion=completion,
                full_text=f"{prompt.text}{completion}",
                num_prompt_tokens=len(prompt.text.split()),
                num_completion_tokens=len(completion.split()),
                finish_reason="mock",
                generation_metadata={
                    "model_name_or_path": "mock",
                    "generation_config": {"backend": "mock", "mode": mode},
                },
            )
        )
    return completions


def mock_completion_text(example: WhetstoneExample, *, mode: str) -> str:
    """Return the stand-in completion text for one example under ``mode``."""
    if mode == "empty":
        return ""
    if example.domain == "math":
        if example.final_answer:
            return f"We compute the result directly. Therefore, \\boxed{{{example.final_answer}}}"
        return "No final answer given."
    if example.domain == "code":
        return example.reference_solution or ""
    return ""
