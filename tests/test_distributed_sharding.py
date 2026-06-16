from whetstone.distributed.sharding import merge_jsonl_files, shard_sequence
from whetstone.utils.jsonl import read_jsonl_list, write_jsonl


def test_strided_sharding_has_no_duplicates_or_missing_items() -> None:
    items = list(range(10))
    rank0 = shard_sequence(items, rank=0, world_size=2)
    rank1 = shard_sequence(items, rank=1, world_size=2)
    assert sorted(rank0 + rank1) == items
    assert set(rank0).isdisjoint(rank1)


def test_strided_sharding_three_ranks() -> None:
    items = list(range(11))
    shards = [shard_sequence(items, rank=rank, world_size=3) for rank in range(3)]
    merged = [item for shard in shards for item in shard]
    assert sorted(merged) == items


def test_merge_jsonl_can_restore_original_order_for_strided_shards(tmp_path) -> None:
    rank0_path = tmp_path / "rank_000_predictions.jsonl"
    rank1_path = tmp_path / "rank_001_predictions.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl([{"uid": "0", "row_index": 0}, {"uid": "2", "row_index": 2}], rank0_path)
    write_jsonl([{"uid": "1", "row_index": 1}, {"uid": "3", "row_index": 3}], rank1_path)

    merge_jsonl_files(
        [rank0_path, rank1_path],
        output_path,
        sort_key=lambda row: int(row["row_index"]),
    )

    assert [row["uid"] for row in read_jsonl_list(output_path)] == ["0", "1", "2", "3"]
