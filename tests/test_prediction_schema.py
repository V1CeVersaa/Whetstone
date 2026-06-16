from whetstone.core.types import (
    ModelCompletion,
    PredictionRecord,
    RenderedPrompt,
    VerificationResult,
    WhetstoneExample,
)
from whetstone.utils.jsonl import read_jsonl_list, write_jsonl


def make_record() -> PredictionRecord:
    example = WhetstoneExample(
        uid="u1",
        domain="math",
        source="fixture",
        split="test",
        prompt_raw="1+1?",
        final_answer="2",
    )
    prompt = RenderedPrompt(uid="u1", template_id="math_cot_boxed_v1", text="Question:\n1+1?")
    completion = ModelCompletion(
        uid="u1",
        completion=r"\boxed{2}",
        full_text=r"Question:\n1+1?\boxed{2}",
        num_prompt_tokens=3,
        num_completion_tokens=2,
    )
    verification = VerificationResult(
        uid="u1",
        domain="math",
        passed=True,
        reward=1.0,
        score=1.0,
        reason="correct",
        extracted_answer="2",
    )
    return PredictionRecord(
        example=example,
        rendered_prompt=prompt,
        completion=completion,
        verification=verification,
    )


def test_prediction_record_round_trip() -> None:
    record = make_record()
    restored = PredictionRecord.from_dict(record.to_dict())
    assert restored.example.uid == record.example.uid
    assert restored.verification.reason == "correct"


def test_prediction_record_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    record = make_record()
    write_jsonl([record.to_flat_dict()], path)
    rows = read_jsonl_list(path)
    assert rows[0]["schema_version"] == "prediction_v1"
    assert rows[0]["uid"] == "u1"
    assert rows[0]["template_id"] == "math_cot_boxed_v1"
    assert rows[0]["passed"] is True
