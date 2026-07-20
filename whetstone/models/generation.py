import re
from collections.abc import Mapping
from contextlib import contextmanager
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

    Prompts are processed in chunks of ``generation_config["batch_size"]``
    (all at once when unset). Chunking bounds the KV cache: hundreds of
    sequences in a single ``generate`` call cost tens of GiB of cache and OOM,
    while per-chunk left padding also wastes less on length variance. Within a
    chunk, generated tokens for every sequence begin at the same padded prompt
    width; token counts exclude padding and ``finish_reason`` reflects whether
    an EOS token was produced.

    Args:
        model: A loaded causal LM.
        tokenizer: Its tokenizer, configured for left-padded generation.
        prompts: Rendered prompts to complete.
        generation_config: Decoding kwargs passed to ``model.generate``
            (``backend`` and ``batch_size`` keys are consumed here, not
            forwarded).
        model_name_or_path: Recorded in each completion's metadata.
        device: Device to move inputs to; ``None``/``"auto"`` leaves them as-is.

    Returns:
        One :class:`ModelCompletion` per prompt, in order.
    """
    batch_size = int(generation_config.get("batch_size") or len(prompts)) or 1
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if "stop_strings" in generation_config and pad_id == eos_id and batch_size > 1:
        # Once several rows share a generated tensor, stop-string padding is
        # indistinguishable from a sampled EOS when pad==eos. Single-row calls
        # have no cross-row padding and therefore preserve the true span.
        logger.warning(
            "stop_strings with pad_token_id == eos_token_id requires batch_size=1 "
            "for unambiguous completion lengths; overriding the requested batch size"
        )
        batch_size = 1
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    logger.info(
        f"Generating {len(prompts)} completions in {num_batches} batch(es) "
        f"(max_new_tokens={generation_config.get('max_new_tokens')}, batch_size={batch_size})"
    )
    completions: list[ModelCompletion] = []
    for batch_index, start in enumerate(range(0, len(prompts), batch_size), start=1):
        completions.extend(
            _generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts[start : start + batch_size],
                generation_config=generation_config,
                model_name_or_path=model_name_or_path,
                device=device,
            )
        )
        logger.info(
            f"Generated {len(completions)}/{len(prompts)} completions "
            f"(batch {batch_index}/{num_batches})"
        )
    return completions


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[RenderedPrompt],
    generation_config: Mapping[str, Any],
    model_name_or_path: str,
    device: str | None,
) -> list[ModelCompletion]:
    """Run one left-padded ``model.generate`` call over a chunk of prompts."""
    prompt_texts = [prompt.text for prompt in prompts]
    encoded = tokenizer(prompt_texts, padding=True, return_tensors="pt")
    if device is not None and device != "auto":
        encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

    gen_kwargs = model_generate_kwargs(generation_config)
    # Pass pad_token_id explicitly: the tokenizer already has one (loader sets
    # pad=eos when absent), and being explicit stops `generate` from emitting a
    # "Setting pad_token_id to eos_token_id" advisory once per call.
    if tokenizer.pad_token_id is not None:
        gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    # HF `generate` needs the tokenizer to honor stop_strings; used by few-shot
    # templates to stop the model from hallucinating further Q/A pairs (whose
    # extra boxed answers would fool the take-last answer extraction).
    if "stop_strings" in gen_kwargs:
        gen_kwargs.setdefault("tokenizer", tokenizer)
    with generation_model_context(model), torch.no_grad():
        generated = model.generate(**encoded, **gen_kwargs)
    if len(generated) != len(prompts):
        msg = (
            f"Expected one generated sequence per prompt, got {len(generated)} "
            f"for {len(prompts)} prompts (num_return_sequences > 1 is unsupported)"
        )
        raise ValueError(msg)
    completions: list[ModelCompletion] = []

    eos_id = tokenizer.eos_token_id
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_token = getattr(tokenizer, "pad_token", None)
    for index, prompt in enumerate(prompts):
        output_ids = generated[index]
        full_text = compact_pad_runs(
            tokenizer.decode(output_ids, skip_special_tokens=True), pad_token
        )

        raw_completion_ids = output_ids[prompt_width:]
        end, stop_kind = true_completion_span(raw_completion_ids, eos_id=eos_id, pad_id=pad_id)
        completion_ids = raw_completion_ids[:end]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        if stop_kind == "eos":
            finish_reason = "stop"
        elif stop_kind == "pad":
            # No EOS but right padding: the row was ended early, e.g. by a
            # configured stop string.
            finish_reason = "stop_string" if "stop_strings" in generation_config else "stop"
        elif matched_stop_string(completion_text, generation_config):
            finish_reason = "stop_string"
        else:
            finish_reason = "length"

        completions.append(
            ModelCompletion(
                uid=prompt.uid,
                completion=completion_text,
                full_text=full_text,
                num_prompt_tokens=int(prompt_lengths[index]),
                num_completion_tokens=end,
                finish_reason=finish_reason,
                generation_metadata={
                    "model_name_or_path": model_name_or_path,
                    "generation_config": dict(generation_config),
                    "tokenizer_ids": {
                        "eos_token_id": eos_id,
                        "pad_token_id": pad_id,
                    },
                },
            )
        )
    return completions


def true_completion_span(
    completion_ids: torch.Tensor,
    *,
    eos_id: int | None,
    pad_id: int | None,
) -> tuple[int, str | None]:
    """Locate the true end of a generated completion inside a padded row.

    Returns ``(end_index, stop_kind)`` where ``completion_ids[:end_index]`` is
    the real generation. Priority: first EOS (kept inclusive -- the stop
    decision is a sampled token); else, when ``pad_id`` differs from
    ``eos_id``, the first pad token (exclusive -- padding appears when HF ends
    a row early without EOS, e.g. via stop_strings, and must never be counted
    or trained on); else the full row (ran to the length limit).

    ``stop_kind`` is ``"eos"``, ``"pad"``, or ``None`` respectively. When
    ``pad_id == eos_id`` the ids alone cannot distinguish stop-string padding
    from a sampled EOS; callers must avoid multi-row padding in that case.
    """
    if eos_id is not None:
        eos_hits = (completion_ids == eos_id).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            return int(eos_hits[0].item()) + 1, "eos"
    if pad_id is not None and pad_id != eos_id:
        pad_hits = (completion_ids == pad_id).nonzero(as_tuple=True)[0]
        if pad_hits.numel() > 0:
            return int(pad_hits[0].item()), "pad"
    return int(completion_ids.numel()), None


def compact_pad_runs(text: str, pad_token: str | None) -> str:
    """Collapse literal pad-token runs in decoded text to ``<pad>*N``.

    ``full_text`` is provenance-only and never re-tokenized, but tokenizers
    whose pad token is not flagged special (e.g. Qwen's ``<|fim_pad|>``) leak
    it through ``skip_special_tokens=True`` decoding, bloating artifacts with
    padding walls. The compact form keeps the structural information (which
    side, how much padding) at a fraction of the size. Runs of one stay
    literal.
    """
    if not pad_token or pad_token not in text:
        return text
    escaped = re.escape(pad_token)
    return re.sub(
        f"(?:{escaped}){{2,}}",
        lambda match: f"{pad_token}*{len(match.group(0)) // len(pad_token)}",
        text,
    )


def matched_stop_string(
    completion_text: str,
    generation_config: Mapping[str, Any],
) -> bool:
    """Return whether decoded text ends with one configured stop string."""
    stop_strings = generation_config.get("stop_strings")
    if isinstance(stop_strings, str):
        stop_strings = [stop_strings]
    if not isinstance(stop_strings, list | tuple):
        return False
    return any(
        isinstance(stop, str) and bool(stop) and completion_text.endswith(stop)
        for stop in stop_strings
    )


@contextmanager
def generation_model_context(model: Any):
    """Temporarily enable eval-mode KV caching and restore model state exactly.

    Gradient checkpointing commonly sets ``model.config.use_cache=False`` for
    training forwards. Generation needs the cache back, but the training
    setting must be restored before the next update.
    """
    was_training = bool(getattr(model, "training", False))
    config = getattr(model, "config", None)
    had_use_cache = config is not None and hasattr(config, "use_cache")
    old_use_cache = getattr(config, "use_cache", None) if had_use_cache else None
    if hasattr(model, "eval"):
        model.eval()
    if had_use_cache:
        config.use_cache = True
    try:
        yield
    finally:
        if had_use_cache:
            config.use_cache = old_use_cache
        if was_training and hasattr(model, "train"):
            model.train()


def model_generate_kwargs(generation_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return kwargs for ``model.generate`` while preserving artifact config elsewhere.

    ``backend`` and ``batch_size`` are Whetstone-level settings consumed before
    generation; ``model.generate`` would reject them as unknown kwargs.
    """
    kwargs = dict(generation_config)
    kwargs.pop("backend", None)
    kwargs.pop("batch_size", None)
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
