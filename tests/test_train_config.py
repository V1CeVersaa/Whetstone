from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from whetstone.core.config import parse_set_overrides, put_dotted_override
from whetstone.train.config import (
    MathRLConfig,
    RLParams,
    SFTTrainConfig,
    TrainEvalConfig,
    TrainingParams,
    load_math_rl_config,
)
from whetstone.train.loop import (
    ensure_single_process_training,
    stage_checkpoint_eval,
    validate_train_references,
)

MATH_RL_BASE = {
    "dataset": {"name": "gsm8k", "split": "train", "limit": 8},
    "prompt": {"template_id": "math_cot_boxed_v1"},
    "model": {"policy_name_or_path": "runs/sft/checkpoints/last"},
    "generation": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 64,
    },
    "verifier": {"name": "math_verify"},
    "rl": {"group_size": 4, "max_seq_length": 128, "kl_beta": 0.0},
}


def test_math_rl_valid_contract_parses() -> None:
    config = MathRLConfig.model_validate(MATH_RL_BASE)
    assert config.rl.group_size == 4
    assert config.reference_path is None


def test_phase1_rejects_nonzero_kl_beta() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 0"):
        RLParams(kl_beta=0.01)


def test_math_rl_rejects_greedy_group_generation() -> None:
    bad = {**MATH_RL_BASE, "generation": {"do_sample": False, "max_new_tokens": 64}}
    with pytest.raises(ValidationError, match="do_sample"):
        MathRLConfig.model_validate(bad)


def test_math_rl_rejects_generation_that_leaves_no_prompt_budget() -> None:
    bad = {
        **MATH_RL_BASE,
        "generation": {"do_sample": True, "temperature": 0.7, "max_new_tokens": 128},
    }
    with pytest.raises(ValidationError, match="max_new_tokens"):
        MathRLConfig.model_validate(bad)


def test_math_rl_rejects_non_math_verifier() -> None:
    bad = {**MATH_RL_BASE, "verifier": {"name": "code_exec"}}
    with pytest.raises(ValidationError, match="math_verify"):
        MathRLConfig.model_validate(bad)


def test_math_rl_rejects_code_dataset_before_loading() -> None:
    bad = {**MATH_RL_BASE, "dataset": {"name": "taco_cobalt", "split": "train"}}
    with pytest.raises(ValidationError, match="domain='code'"):
        MathRLConfig.model_validate(bad)


def test_math_rl_rejects_protected_test_split() -> None:
    bad = {**MATH_RL_BASE, "dataset": {"name": "gsm8k", "split": "test"}}
    with pytest.raises(ValidationError, match="protected"):
        MathRLConfig.model_validate(bad)


def test_sft_rejects_protected_test_split() -> None:
    bad = {
        "dataset": {"name": "openr1_math", "split": "test", "limit": 8},
        "prompt": {"template_id": "math_cot_boxed_v1"},
        "model": {"name_or_path": "Qwen/Qwen3-0.6B-Base"},
        "training": {"max_steps": 1},
    }
    with pytest.raises(ValidationError, match="protected"):
        SFTTrainConfig.model_validate(bad)


def test_runtime_reference_validation_rejects_wrong_domain() -> None:
    with pytest.raises(ValueError, match="expected 'math'"):
        validate_train_references("tiny_code", "code_python_solution_v1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("gradient_accumulation_steps", 0),
        ("learning_rate", 0.0),
        ("max_seq_length", 1),
        ("log_every", 0),
    ],
)
def test_sft_training_rejects_nonpositive_runtime_values(field: str, value: int | float) -> None:
    with pytest.raises(ValidationError):
        TrainingParams(max_steps=1, **{field: value})


def test_sft_step_budgets_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="alternatives"):
        TrainingParams(max_steps=1, num_epochs=1)


def test_training_rejects_torchrun_environment(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="single-process only"):
        ensure_single_process_training(False)


def test_checkpoint_eval_is_staged_without_running_evaluator(tmp_path) -> None:
    eval_config = TrainEvalConfig.model_validate(
        {
            "dataset": {"name": "gsm8k", "split": "test", "limit": 4},
            "prompt": {"template_id": "math_cot_boxed_v1"},
            "verifier": {"name": "math_verify"},
            "generation": {"max_new_tokens": 32},
        }
    )
    config_path = stage_checkpoint_eval(
        eval_config,
        run_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints" / "step_000001",
        step=1,
        seed=42,
        device="cuda",
        dtype="bf16",
    )

    assert config_path == tmp_path / "eval" / "step_000001" / "eval_config.yaml"
    assert config_path.exists()
    assert not (config_path.parent / "predictions.jsonl").exists()


def test_cli_overrides_beat_config_files_before_validation(tmp_path) -> None:
    config_file = tmp_path / "rl.yaml"
    config_file.write_text(yaml.safe_dump(MATH_RL_BASE), encoding="utf-8")

    config = load_math_rl_config(
        config_file,
        overrides=parse_set_overrides(
            [
                "model.policy_name_or_path=runs/other_sft/checkpoints/last",
                "rl.max_steps=5",
                "rl.eval_every=null",
            ]
        ),
    )

    assert config.model.policy_name_or_path == "runs/other_sft/checkpoints/last"
    assert config.rl.max_steps == 5
    assert config.rl.eval_every is None


def test_committed_math_rl_smoke_requires_explicit_policy_override() -> None:
    config_path = Path("configs/train/math_rl_smoke.yaml")
    with pytest.raises(ValidationError, match="policy_name_or_path"):
        load_math_rl_config(config_path)

    config = load_math_rl_config(
        config_path,
        overrides={"model": {"policy_name_or_path": "runs/sft/checkpoints/last"}},
    )
    assert config.model.policy_name_or_path == "runs/sft/checkpoints/last"


def test_typo_in_override_key_is_rejected_by_strict_schema(tmp_path) -> None:
    config_file = tmp_path / "rl.yaml"
    config_file.write_text(yaml.safe_dump(MATH_RL_BASE), encoding="utf-8")

    with pytest.raises(ValidationError, match="max_step"):
        load_math_rl_config(config_file, overrides=parse_set_overrides(["rl.max_step=5"]))


def test_dotted_override_shape_conflicts_are_rejected() -> None:
    overrides: dict = {}
    put_dotted_override(overrides, "model.policy_name_or_path", "x")
    with pytest.raises(ValueError, match="conflicts with scalar"):
        put_dotted_override(overrides, "model.policy_name_or_path.deeper", 1)
    with pytest.raises(ValueError, match="replace section"):
        put_dotted_override(overrides, "model", "scalar")
    with pytest.raises(ValueError, match="empty path segment"):
        put_dotted_override(overrides, "rl..max_steps", 5)


def test_parse_set_overrides_yaml_values_and_shape() -> None:
    overrides = parse_set_overrides(["rl.max_steps=5", "rl.eval_every=null", "run.name=abc"])
    assert overrides == {
        "rl": {"max_steps": 5, "eval_every": None},
        "run": {"name": "abc"},
    }
    with pytest.raises(ValueError, match=r"SECTION\.KEY=VALUE"):
        parse_set_overrides(["rl.max_steps"])
