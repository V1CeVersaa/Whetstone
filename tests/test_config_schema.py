import pytest
from pydantic import ValidationError

from whetstone.core.config import EvalConfig, load_eval_config, merge_config_files
from whetstone.eval.runner import validate_config_references

BASE = {
    "run": {"name": "t", "seed": 1},
    "dataset": {"name": "gsm8k", "split": "test", "limit": 5},
    "prompt": {"template_id": "math_cot_boxed_v1"},
    "model": {"name_or_path": "Qwen/Qwen3-0.6B-Base", "device": "cuda"},
    "verifier": {"name": "math_verify"},
}


def test_valid_config_parses_and_fills_defaults() -> None:
    config = EvalConfig.model_validate(BASE)
    assert config.dataset.limit == 5
    assert config.generation.max_new_tokens == 512  # default filled in
    assert config.device == "cuda"
    assert config.effective_backend == "transformers"


def test_unknown_top_level_key_is_rejected() -> None:
    bad = {**BASE, "generaton": {"max_new_tokens": 8}}  # typo'd block
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(bad)


def test_unknown_nested_key_is_rejected() -> None:
    bad = {**BASE, "dataset": {"name": "gsm8k", "splt": "test"}}  # typo'd field
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(bad)


def test_missing_required_field_is_rejected() -> None:
    bad = {k: v for k, v in BASE.items() if k != "dataset"}
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(bad)


def test_unknown_verifier_name_is_rejected() -> None:
    bad = {**BASE, "verifier": {"name": "made_up"}}
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(bad)


def test_negative_temperature_is_rejected() -> None:
    bad = {**BASE, "generation": {"temperature": -1.0}}
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(bad)


def test_sampling_requires_positive_temperature() -> None:
    bad = {**BASE, "generation": {"do_sample": True, "temperature": 0.0}}
    with pytest.raises(ValidationError, match="temperature"):
        EvalConfig.model_validate(bad)


def test_top_p_zero_is_rejected() -> None:
    bad = {**BASE, "generation": {"top_p": 0.0}}
    with pytest.raises(ValidationError, match="top_p"):
        EvalConfig.model_validate(bad)


def test_zero_dataset_limit_is_rejected() -> None:
    bad = {**BASE, "dataset": {"name": "gsm8k", "split": "test", "limit": 0}}
    with pytest.raises(ValidationError, match="limit"):
        EvalConfig.model_validate(bad)


def test_multiple_return_sequences_are_rejected_by_eval_schema() -> None:
    bad = {**BASE, "generation": {"num_return_sequences": 2}}
    with pytest.raises(ValidationError, match="num_return_sequences"):
        EvalConfig.model_validate(bad)


def test_nonpositive_verifier_limits_are_rejected() -> None:
    bad = {**BASE, "verifier": {"name": "math_verify", "max_chars": 0}}
    with pytest.raises(ValidationError, match="max_chars"):
        EvalConfig.model_validate(bad)


def test_mock_backend_does_not_require_model_name() -> None:
    mock = {**BASE, "model": {"backend": "mock", "mock_mode": "reference"}}
    config = EvalConfig.model_validate(mock)
    assert config.effective_backend == "mock"
    assert config.model.name_or_path is None


def test_real_backend_requires_model_name() -> None:
    bad = {**BASE, "model": {"device": "cuda"}}  # no name_or_path, not mock
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(bad)


def test_generation_allows_extra_hf_kwargs() -> None:
    config = EvalConfig.model_validate(
        {**BASE, "generation": {"num_return_sequences": 1, "repetition_penalty": 1.1}}
    )
    dumped = config.generation.model_dump()
    assert dumped["num_return_sequences"] == 1
    assert dumped["repetition_penalty"] == 1.1


def test_runtime_device_overrides_model_device() -> None:
    config = EvalConfig.model_validate({**BASE, "runtime": {"device": "cpu"}})
    assert config.device == "cpu"


def test_unknown_dataset_reference_is_rejected_before_runner_work() -> None:
    config = EvalConfig.model_validate({**BASE, "dataset": {"name": "made_up", "split": "test"}})
    with pytest.raises(ValueError, match=r"dataset\.name"):
        validate_config_references(config)


def test_unknown_prompt_reference_is_rejected_before_runner_work() -> None:
    config = EvalConfig.model_validate({**BASE, "prompt": {"template_id": "made_up_v1"}})
    with pytest.raises(ValueError, match=r"prompt\.template_id"):
        validate_config_references(config)


def test_merge_layers_runtime_over_base(tmp_path) -> None:
    import yaml

    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(yaml.safe_dump({"runtime": {"device": "cpu"}}), encoding="utf-8")

    merged = merge_config_files([base_path, runtime_path])
    assert merged["runtime"]["device"] == "cpu"
    assert merged["dataset"]["name"] == "gsm8k"  # base survives

    config = load_eval_config([base_path, runtime_path])
    assert config.device == "cpu"
