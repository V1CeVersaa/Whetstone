import json

import pytest
import yaml

from whetstone.eval.runner import run_evaluation
from whetstone.utils.jsonl import read_jsonl_list


def test_tiny_math_mock_runner_writes_complete_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "tiny_math_run"
    config_path = tmp_path / "tiny_math_mock.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "tiny_math_test", "output_dir": str(run_dir), "seed": 123},
                "dataset": {"name": "tiny_math", "split": "validation", "limit": 2},
                "prompt": {"template_id": "math_cot_boxed_v1"},
                "model": {"backend": "mock", "mock_mode": "reference"},
                "generation": {"backend": "mock"},
                "verifier": {"name": "math_answer"},
                "runtime": {"distributed": False},
            }
        ),
        encoding="utf-8",
    )

    returned_run_dir = run_evaluation(config_path)

    assert returned_run_dir == run_dir
    for filename in (
        "run_config.yaml",
        "env.json",
        "predictions.jsonl",
        "metrics.json",
        "samples.md",
        "status.json",
    ):
        assert (run_dir / filename).exists()
    rows = read_jsonl_list(run_dir / "predictions.jsonl")
    assert len(rows) == 2
    assert rows[0]["schema_version"] == "prediction_v1"
    assert rows[0]["source"] == "tiny_math"
    assert rows[0]["template_id"] == "math_cot_boxed_v1"
    assert rows[0]["prompt"]
    assert rows[0]["completion"]
    assert rows[0]["reason"] == "correct"
    assert rows[0]["diagnostics"]["verifier"]["name"] == "math_answer"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_examples"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["world_size"] == 1

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["num_predictions"] == 2


def test_runner_writes_failed_status_after_run_dir_creation(tmp_path) -> None:
    run_dir = tmp_path / "failed_run"
    config_path = tmp_path / "bad_domain.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "bad_domain", "output_dir": str(run_dir), "seed": 123},
                "dataset": {"name": "tiny_code", "split": "validation", "limit": 1},
                "prompt": {"template_id": "math_cot_boxed_v1"},
                "model": {"backend": "mock", "mock_mode": "reference"},
                "generation": {"backend": "mock"},
                "verifier": {"name": "code_exec"},
                "runtime": {"distributed": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expects domain"):
        run_evaluation(config_path)

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["stage"] == "prompt_rendering"
    assert status["error_type"] == "ValueError"
