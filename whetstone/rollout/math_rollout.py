from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from whetstone.core.types import ModelCompletion, RenderedPrompt, WhetstoneExample
from whetstone.models.generation import model_generate_kwargs
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
    """Sample grouped completions with ``model.generate`` and slice out token ids."""
    logger.debug(
        f"Rollout generation: {len(prompt_texts)} prompt(s) x {group_size} completions "
        f"(max_new_tokens={generation_config.get('max_new_tokens')})"
    )
    encoded = tokenizer(prompt_texts, padding=True, return_tensors="pt")
    if device is not None and device != "auto":
        encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    gen_kwargs = model_generate_kwargs(generation_config)
    gen_kwargs["num_return_sequences"] = group_size
    if "stop_strings" in gen_kwargs:
        gen_kwargs.setdefault("tokenizer", tokenizer)
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        with torch.no_grad():
            generated = model.generate(**encoded, **gen_kwargs)
    finally:
        if was_training:
            model.train()

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

    eos_id = tokenizer.eos_token_id
    completion_ids_per_sample: list[list[int]] = []
    completion_texts: list[str] = []
    for row in range(expected):
        completion = generated[row][prompt_width:]
        # Keep tokens up to and including the first EOS; the rest is padding.
        if eos_id is not None:
            eos_hits = (completion == eos_id).nonzero(as_tuple=True)[0]
            if eos_hits.numel() > 0:
                completion = completion[: int(eos_hits[0].item()) + 1]
        ids = completion.tolist()
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
