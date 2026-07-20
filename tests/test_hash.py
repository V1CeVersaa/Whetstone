from whetstone.utils.hash import git_source_state, source_tree_hash


def test_source_tree_hash_covers_untracked_runtime_sources(tmp_path) -> None:
    source_dir = tmp_path / "whetstone"
    source_dir.mkdir()
    source_file = source_dir / "module.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")

    first_hash, first_count = source_tree_hash(tmp_path)
    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    second_hash, second_count = source_tree_hash(tmp_path)

    assert first_count == second_count == 1
    assert first_hash != second_hash


def test_git_source_state_has_reproducibility_fields_without_git_repo(tmp_path) -> None:
    source_dir = tmp_path / "scripts"
    source_dir.mkdir()
    (source_dir / "run.py").write_text("print('ok')\n", encoding="utf-8")

    state = git_source_state(tmp_path)

    assert state["git_commit"] is None
    assert state["git_dirty"] is False
    assert state["source_dirty"] is False
    assert state["source_file_count"] == 1
    assert len(state["source_tree_sha256"]) == 64
    assert state["source_scope"]
