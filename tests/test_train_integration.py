"""Offline integration tests: tiny model, fake tokenizer, no downloads.

The SFT test is the mandatory overfit check scaled down: if a 2-layer model
cannot drive its loss down on three fixture examples, the loop (masking,
loss, optimizer plumbing) is broken. The RL test drives the full rollout ->
verify -> advantage -> update -> artifact path with injected completions.
"""

import copy
import json

import torch
from transformers import GPT2LMHeadModel

from whetstone.core.config import VerifierConfig
from whetstone.data.tiny import TinyMathAdapter
from whetstone.prompts.templates import render_prompts
from whetstone.train.config import AdvantageConfig, RLParams, TrainingParams
from whetstone.train.examples import build_sft_examples
from whetstone.utils.jsonl import read_jsonl_list
from whetstone.verify import build_verifier


def load_metrics_rows(run_dir):
    return read_jsonl_list(run_dir / "metrics.jsonl")


def test_sft_overfit_tiny_model_and_artifacts(tmp_path, tiny_causal_lm, word_tokenizer) -> None:
    examples = TinyMathAdapter().load(split="train")
    sft_examples, _ = build_sft_examples(examples, "math_cot_boxed_v1")
    assert len(sft_examples) == 3

    from whetstone.algorithms.sft import train_sft_loop

    torch.manual_seed(0)
    params = TrainingParams(
        max_steps=40,
        batch_size=3,
        gradient_accumulation_steps=1,
        learning_rate=1.0e-2,
        max_grad_norm=1.0,
        log_every=1,
        save_every=None,
        eval_every=None,
        max_seq_length=128,
    )
    run_dir = tmp_path / "sft_run"
    run_dir.mkdir()
    final_metrics = train_sft_loop(
        model=tiny_causal_lm,
        tokenizer=word_tokenizer,
        sft_examples=sft_examples,
        params=params,
        run_dir=run_dir,
        device="cpu",
        seed=42,
    )

    # The overfit criterion: loss must clearly decrease on a tiny subset.
    assert final_metrics["final_train_loss"] < final_metrics["first_train_loss"] * 0.8

    rows = load_metrics_rows(run_dir)
    assert len(rows) == params.max_steps
    assert {"step", "train_loss", "learning_rate", "step_time"} <= set(rows[0])

    last_dir = run_dir / "checkpoints" / "last"
    assert (last_dir / "training_state.json").exists()
    state = json.loads((last_dir / "training_state.json").read_text(encoding="utf-8"))
    assert state["step"] == params.max_steps

    # The checkpoint must be loadable the way the Foundation eval runner loads models.
    reloaded = GPT2LMHeadModel.from_pretrained(last_dir)
    assert reloaded.config.vocab_size == tiny_causal_lm.config.vocab_size


def test_math_rl_tiny_loop_writes_rollout_artifacts(
    tmp_path, tiny_causal_lm, word_tokenizer
) -> None:
    examples = TinyMathAdapter().load(split="train")
    rendered = render_prompts(examples, "math_cot_boxed_v1")
    gold_by_prompt = {
        prompt.text: example.final_answer for prompt, example in zip(rendered, examples, strict=True)
    }

    def generate_fn(prompt_texts, group_size):
        # One correct and one wrong completion per prompt -> every group has
        # reward variance, so every step carries a learning signal.
        groups = []
        for text in prompt_texts:
            gold = gold_by_prompt[text]
            groups.append(
                [
                    f"The final answer is \\boxed{{{gold}}}",
                    "The final answer is \\boxed{999999}",
                ][:group_size]
            )
        return groups

    from whetstone.algorithms.math_rl import train_math_rl_loop

    torch.manual_seed(0)
    rl = RLParams(
        group_size=2,
        prompts_per_step=2,
        max_steps=3,
        learning_rate=1.0e-3,
        kl_beta=0.0,
        log_every=1,
        save_every=None,
        eval_every=None,
        max_seq_length=128,
    )
    run_dir = tmp_path / "rl_run"
    run_dir.mkdir()
    final_metrics = train_math_rl_loop(
        policy=tiny_causal_lm,
        reference=None,
        tokenizer=word_tokenizer,
        examples=examples,
        rendered_prompts=rendered,
        verifier=build_verifier(VerifierConfig(name="math_answer")),
        rl=rl,
        advantage=AdvantageConfig(),
        generation_config={"max_new_tokens": 8, "do_sample": True},
        run_dir=run_dir,
        device="cpu",
        seed=42,
        generate_fn=generate_fn,
    )

    rollout_rows = read_jsonl_list(run_dir / "rollout_samples.jsonl")
    assert rollout_rows
    assert final_metrics["num_rollout_samples"] == len(rollout_rows)
    assert {row["step"] for row in rollout_rows} == {1, 2, 3}
    required_fields = {
        "uid",
        "group_id",
        "prompt_text",
        "completion_text",
        "reward",
        "passed",
        "verifier_reason",
        "advantage",
        "num_completion_tokens",
    }
    assert required_fields <= set(rollout_rows[0])
    # Exactly one of each group's two completions is correct.
    assert {row["verifier_reason"] for row in rollout_rows} == {"correct", "wrong_answer"}
    assert final_metrics["mean_reward_overall"] == 0.5

    metrics_rows = load_metrics_rows(run_dir)
    assert len(metrics_rows) == rl.max_steps
    first = metrics_rows[0]
    assert first["mean_reward"] == 0.5
    assert first["nonzero_group_variance_rate"] == 1.0
    assert {"rl_loss", "policy_loss", "kl", "rollout_time", "update_time"} <= set(first)
    # Group-mean advantages for [1, 0] rewards.
    step_one = [row["advantage"] for row in rollout_rows if row["step"] == 1]
    assert sorted(step_one) == sorted([0.5, -0.5] * (len(step_one) // 2))

    assert (run_dir / "checkpoints" / "last" / "training_state.json").exists()
    assert (run_dir / "samples.md").exists()


def test_math_rl_kl_path_runs_with_frozen_reference(
    tmp_path, tiny_causal_lm, word_tokenizer
) -> None:
    examples = TinyMathAdapter().load(split="train")
    rendered = render_prompts(examples, "math_cot_boxed_v1")

    def generate_fn(prompt_texts, group_size):
        return [["\\boxed{1}", "\\boxed{2}"][:group_size] for _ in prompt_texts]

    from whetstone.algorithms.math_rl import train_math_rl_loop

    reference = copy.deepcopy(tiny_causal_lm)
    reference.requires_grad_(False)

    rl = RLParams(
        group_size=2,
        prompts_per_step=1,
        max_steps=2,
        learning_rate=1.0e-3,
        kl_beta=0.1,
        log_every=1,
        save_every=None,
        eval_every=None,
        max_seq_length=128,
        # Exercise the chunked update path: per-chunk reweighting must still
        # reproduce full-batch normalization (kl stays exactly 0 at step 1).
        update_micro_batch_size=1,
    )
    run_dir = tmp_path / "rl_kl_run"
    run_dir.mkdir()
    train_math_rl_loop(
        policy=tiny_causal_lm,
        reference=reference,
        tokenizer=word_tokenizer,
        examples=examples,
        rendered_prompts=rendered,
        verifier=build_verifier(VerifierConfig(name="math_answer")),
        rl=rl,
        advantage=AdvantageConfig(),
        generation_config={"max_new_tokens": 8, "do_sample": True},
        run_dir=run_dir,
        device="cpu",
        seed=42,
        generate_fn=generate_fn,
    )
    rows = load_metrics_rows(run_dir)
    assert all(isinstance(row["kl"], float) for row in rows)
    # Step 1 runs before any update, so policy == reference and the sampled KL
    # estimate is exactly zero.
    assert rows[0]["kl"] == 0.0
