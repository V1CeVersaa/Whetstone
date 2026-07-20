from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from whetstone.core.types import ModelCompletion, RenderedPrompt, WhetstoneExample
from whetstone.models.generation import (
    generation_model_context,
    model_generate_kwargs,
    true_completion_span,
)
from whetstone.train.types import MathRolloutSample
from whetstone.utils.logging import get_logger
from whetstone.verify.base import Verifier

logger = get_logger(__name__)

# Test-injectable generator: (prompt_texts, group_size) -> one list of G
# completion strings per prompt. Lets RL loop tests run without model.generate.
GenerateFn = Callable[[list[str], int], list[list[str]]]


@dataclass
class RolloutBatch:
    """One step's grouped rollouts plus the exact token ids for the update pass.

    ``samples[i]`` corresponds to ``prompt_token_ids[i]`` and
    ``completion_token_ids[i]``; entries are ordered group-contiguously
    (samples ``i*G .. (i+1)*G - 1`` share a prompt). Keeping the generated
    token ids (rather than re-tokenizing text) is what guarantees the policy
    logprobs are computed on exactly the sampled tokens.
    """

    samples: list[MathRolloutSample]
    prompt_token_ids: list[list[int]]
    completion_token_ids: list[list[int]]


def generate_grouped_rollouts(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[WhetstoneExample],
    prompts: list[RenderedPrompt],
    group_size: int,
    generation_config: Mapping[str, Any],
    verifier: Verifier,
    device: str | None = None,
    group_prefix: str = "",
    generate_fn: GenerateFn | None = None,
) -> RolloutBatch:
    """Sample ``group_size`` completions per prompt and verify each one.

    Uses ``model.generate`` with ``num_return_sequences=group_size`` (the
    tokenizer must be left-padding, as configured by the Foundation loader).
    ``generate_fn`` swaps in canned completions for offline tests; token ids
    then come from re-tokenizing that text, which is acceptable only for the
    test path.

    Rewards are binary -- ``1.0`` iff the math verifier passed -- with no
    shaping, per Math-RL v0.
    """
    prompt_texts = [prompt.text for prompt in prompts]

    if generate_fn is None:
        prompt_ids_per_prompt, completion_ids_per_sample, completion_texts = _generate_real(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            group_size=group_size,
            generation_config=generation_config,
            device=device,
        )
    else:
        prompt_ids_per_prompt, completion_ids_per_sample, completion_texts = _generate_fake(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            group_size=group_size,
            generate_fn=generate_fn,
        )

    samples: list[MathRolloutSample] = []
    prompt_token_ids: list[list[int]] = []
    for prompt_index, (example, prompt) in enumerate(zip(examples, prompts, strict=True)):
        group_id = f"{group_prefix}{example.uid}"
        for member in range(group_size):
            flat = prompt_index * group_size + member
            completion_text = completion_texts[flat]
            completion_ids = completion_ids_per_sample[flat]
            prompt_ids = prompt_ids_per_prompt[prompt_index]

            verification = verifier.verify(
                example,
                ModelCompletion(
                    uid=example.uid,
                    completion=completion_text,
                    full_text=f"{prompt.text}{completion_text}",
                    num_prompt_tokens=len(prompt_ids),
                    num_completion_tokens=len(completion_ids),
                ),
            )
            samples.append(
                MathRolloutSample(
                    uid=example.uid,
                    group_id=group_id,
                    prompt_text=prompt.text,
                    completion_text=completion_text,
                    reward=1.0 if verification.passed else 0.0,
                    passed=verification.passed,
                    verifier_reason=verification.reason,
                    extracted_answer=verification.extracted_answer,
                    diagnostics=verification.diagnostics,
                    num_prompt_tokens=len(prompt_ids),
                    num_completion_tokens=len(completion_ids),
                    metadata={
                        "gold_answer": example.final_answer,
                        "template_id": prompt.template_id,
                        "group_member": member,
                        "generation_config": dict(generation_config),
                        "tokenizer_ids": {
                            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                        },
                    },
                )
            )
            prompt_token_ids.append(list(prompt_ids))

    return RolloutBatch(
        samples=samples,
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_ids_per_sample,
    )


def _generate_real(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    group_size: int,
    generation_config: Mapping[str, Any],
    device: str | None,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    """Sample grouped completions in bounded prompt batches."""
    batch_size = int(generation_config.get("batch_size") or len(prompt_texts)) or 1
    num_batches = (len(prompt_texts) + batch_size - 1) // batch_size
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if "stop_strings" in generation_config and pad_id == eos_id and group_size > 1:
        msg = (
            "Math-RL stop_strings require pad_token_id != eos_token_id when group_size > 1; "
            "otherwise stop-string padding is indistinguishable from sampled EOS tokens"
        )
        raise ValueError(msg)
    logger.info(
        f"Rollout generation: {len(prompt_texts)} prompt(s) x {group_size} completions "
        f"in {num_batches} batch(es) (prompt_batch_size={batch_size}, "
        f"max_new_tokens={generation_config.get('max_new_tokens')})"
    )

    prompt_ids_per_prompt: list[list[int]] = []
    completion_ids_per_sample: list[list[int]] = []
    completion_texts: list[str] = []
    with generation_model_context(model):
        for start in range(0, len(prompt_texts), batch_size):
            batch_prompt_ids, batch_completion_ids, batch_completion_texts = _generate_real_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts[start : start + batch_size],
                group_size=group_size,
                generation_config=generation_config,
                device=device,
            )
            prompt_ids_per_prompt.extend(batch_prompt_ids)
            completion_ids_per_sample.extend(batch_completion_ids)
            completion_texts.extend(batch_completion_texts)
    return prompt_ids_per_prompt, completion_ids_per_sample, completion_texts


def _generate_real_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    group_size: int,
    generation_config: Mapping[str, Any],
    device: str | None,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    """Run one grouped ``model.generate`` call and keep exact sampled ids."""
    encoded = tokenizer(prompt_texts, padding=True, return_tensors="pt")
    if device is not None and device != "auto":
        encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    gen_kwargs = model_generate_kwargs(generation_config)
    gen_kwargs["num_return_sequences"] = group_size
    if tokenizer.pad_token_id is not None:
        gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    if "stop_strings" in gen_kwargs:
        gen_kwargs.setdefault("tokenizer", tokenizer)
    with torch.no_grad():
        generated = model.generate(**encoded, **gen_kwargs)

    expected = len(prompt_texts) * group_size
    if len(generated) != expected:
        msg = (
            f"Expected {expected} sequences ({len(prompt_texts)} prompts x "
            f"group_size {group_size}), got {len(generated)}"
        )
        raise ValueError(msg)

    # Strip left padding: the attention mask marks each prompt's real tokens.
    prompt_ids_per_prompt = [
        encoded["input_ids"][row][encoded["attention_mask"][row] == 1].tolist()
        for row in range(len(prompt_texts))
    ]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    completion_ids_per_sample: list[list[int]] = []
    completion_texts: list[str] = []
    for row in range(expected):
        completion = generated[row][prompt_width:]
        # Cut at the first EOS (inclusive) or, when pad differs from EOS, at
        # the first pad (exclusive): rows ended early by stop_strings carry no
        # EOS and their padding must never enter completion_ids, or the RL
        # update would train the policy on pad positions.
        end, _ = true_completion_span(completion, eos_id=eos_id, pad_id=pad_id)
        ids = completion[:end].tolist()
        completion_ids_per_sample.append(ids)
        completion_texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return prompt_ids_per_prompt, completion_ids_per_sample, completion_texts


def _generate_fake(
    *,
    tokenizer: Any,
    prompt_texts: list[str],
    group_size: int,
    generate_fn: GenerateFn,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    """Build rollout token ids from injected completion texts (test path only)."""
    grouped_texts = generate_fn(prompt_texts, group_size)
    if len(grouped_texts) != len(prompt_texts) or any(
        len(group) != group_size for group in grouped_texts
    ):
        msg = f"generate_fn must return {len(prompt_texts)} groups of {group_size} completions"
        raise ValueError(msg)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    prompt_ids_per_prompt = [
        list(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in prompt_texts
    ]
    completion_ids_per_sample: list[list[int]] = []
    completion_texts: list[str] = []
    for group in grouped_texts:
        for text in group:
            ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            if eos_id is not None:
                ids.append(eos_id)
            completion_ids_per_sample.append(ids)
            completion_texts.append(text)
    return prompt_ids_per_prompt, completion_ids_per_sample, completion_texts
