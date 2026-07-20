import json
from pathlib import Path

from whetstone.core.config import save_yaml
from whetstone.train.validation import validate_training_run
from whetstone.utils.hash import stable_hash
from whetstone.utils.jsonl import write_jsonl


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_common_artifacts(run_dir: Path, config: dict, final_metrics: dict) -> None:
    run_dir.mkdir()
    save_yaml(config, run_dir / "train_config.yaml")
    write_json(
        run_dir / "env.json",
        {
            "config_sha256": stable_hash(config),
            "packages": {"math-verify": "0.9.0"},
            "source_state": {
                "git_commit": "a" * 40,
                "git_dirty": True,
                "source_dirty": False,
                "source_status": [],
                "source_tree_sha256": "b" * 64,
                "source_file_count": 10,
            },
        },
    )
    write_json(run_dir / "status.json", {"status": "completed", "stage": "training"})
    write_json(run_dir / "final_metrics.json", final_metrics)
    write_jsonl([{"step": 1, "grad_norm": 1.0}], run_dir / "metrics.jsonl")
    checkpoint = run_dir / "checkpoints" / "last"
    checkpoint.mkdir(parents=True)
    for filename in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (checkpoint / filename).write_text("{}", encoding="utf-8")


def test_valid_sft_run_passes_all_gates(tmp_path) -> None:
    run_dir = tmp_path / "sft"
    write_common_artifacts(
        run_dir,
        {
            "preprocessing": {
                "ensure_verifiable_target": True,
                "max_decode_mismatch_rate": 0.0,
            },
            "training": {"max_steps": 1},
        },
        {
            "num_steps": 1,
            "first_train_loss": 1.0,
            "final_train_loss": 0.25,
            "num_training_examples": 2,
            "num_overlong_dropped": 0,
            "peak_gpu_memory_mb": 0.0,
        },
    )
    write_json(
        run_dir / "preprocessing.json",
        {
            "num_loaded_examples": 2,
            "num_sft_examples": 2,
            "target_candidate_verifiable_rate": 1.0,
            "emitted_target_verifier_pass_rate": 1.0,
            "target_initial_parse_failure_rate": 0.0,
            "target_initial_answer_mismatch_rate": 0.0,
            "target_declared_conflict_rate": 0.0,
        },
    )
    write_json(
        run_dir / "tokenization_audit.json",
        {"decode_mismatch_rate": 0.0, "separate_joint_mismatch_rate": 0.0},
    )

    report = validate_training_run(run_dir, require_loss_decrease=True)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["summary"]["loss_decreased"] is True
    assert report["summary"]["git_dirty"] is True
    assert report["summary"]["source_dirty"] is False


def test_rl_run_rejects_broken_group_advantage(tmp_path) -> None:
    run_dir = tmp_path / "rl"
    write_common_artifacts(
        run_dir,
        {"rl": {"max_steps": 1, "prompts_per_step": 1, "group_size": 2}},
        {
            "num_steps": 1,
            "num_rollout_samples": 2,
            "mean_reward_overall": 0.5,
            "peak_gpu_memory_mb": 0.0,
        },
    )
    write_jsonl(
        [
            {
                "step": 1,
                "group_id": "step_1/prompt",
                "reward": 1.0,
                "passed": True,
                "verifier_reason": "correct",
                "advantage": 0.75,
            },
            {
                "step": 1,
                "group_id": "step_1/prompt",
                "reward": 0.0,
                "passed": False,
                "verifier_reason": "wrong_answer",
                "advantage": -0.25,
            },
        ],
        run_dir / "rollout_samples.jsonl",
    )
    write_jsonl(
        [
            {
                "step": 1,
                "grad_norm": 1.0,
                "nonzero_group_variance_rate": 1.0,
                "mean_abs_advantage": 0.5,
                "avg_completion_tokens": 8.0,
                "boxed_completion_rate": 1.0,
            }
        ],
        run_dir / "metrics.jsonl",
    )

    report = validate_training_run(run_dir)

    assert report["valid"] is False
    assert any("advantage sum" in error for error in report["errors"])
