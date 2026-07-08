import json
import platform
import socket
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

from whetstone.core.config import (
    EvalConfig,
    GenerationConfig,
    ModelConfig,
    deep_merge,
    merge_config_files,
    save_yaml,
)
from whetstone.core.paths import ensure_dir, resolve_project_path
from whetstone.core.run_id import make_run_id
from whetstone.core.seed import set_seed
from whetstone.core.types import PredictionRecord
from whetstone.data import get_dataset_adapter
from whetstone.data.base import DATASET_REGISTRY
from whetstone.distributed.init import (
    DistributedState,
    barrier,
    broadcast_object,
    init_distributed,
    shutdown_distributed,
)
from whetstone.distributed.sharding import merge_jsonl_files, shard_sequence
from whetstone.eval.metrics import compute_metrics
from whetstone.eval.prediction_io import read_prediction_rows, write_predictions
from whetstone.eval.samples import write_samples_markdown
from whetstone.models.generation import generate_completions, generate_mock_completions
from whetstone.models.loader import load_causal_lm
from whetstone.prompts.templates import TEMPLATE_REGISTRY, render_prompts
from whetstone.utils.hash import git_commit
from whetstone.utils.logging import configure_logging, get_logger
from whetstone.utils.memory import peak_gpu_memory_mb
from whetstone.verify import VERIFIER_REGISTRY, build_verifier

logger = get_logger(__name__)


def run_evaluation(
    config_path: str | Path | Sequence[str | Path],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Run one end-to-end evaluation and return its run directory.

    Loads and merges the config(s), sets the seed, initializes distributed state,
    then loads and shards the dataset, renders prompts, generates completions,
    verifies them, and writes per-rank predictions. Rank 0 merges predictions and
    writes ``run_config.yaml``, ``env.json``, ``metrics.json``, and ``samples.md``.

    Args:
        config_path: A single config path, or a sequence of paths that are
            deep-merged in order (later ones override earlier ones).
        overrides: Optional nested dict merged on top of the config files
            (used for CLI flags like ``--model``); it goes through the same
            deep-merge as the files, so the saved ``run_config.yaml`` reflects
            exactly what ran.

    Returns:
        Path to the created run directory.
    """
    config_paths: list[Path] = (
        [Path(config_path)]
        if isinstance(config_path, (str, Path))
        else [Path(path) for path in config_path]
    )
    data = merge_config_files(config_paths)
    if overrides:
        data = deep_merge(data, overrides)
    config = EvalConfig.model_validate(data)
    validate_config_references(config)
    config_path = Path(config_paths[0])  # The first config names the run

    device = config.device
    set_seed(config.run.seed)

    state = init_distributed(device)
    configure_logging(rank=state.rank, force=True)
    run_dir = create_run_dir(config, config_path, state)

    stage = "run_metadata"
    start_time = perf_counter()
    try:
        logger.info(
            f"Run {run_dir.name} | dataset={config.dataset.name} split={config.dataset.split} "
            f"limit={config.dataset.limit} template={config.prompt.template_id} verifier={config.verifier.name} "
            f"backend={config.effective_backend} world_size={state.world_size}"
        )

        if state.is_main:
            save_yaml(config, run_dir / "run_config.yaml")
            write_json(run_dir / "env.json", collect_env(state))
            write_status(run_dir, status="running", stage=stage)
        barrier(state)

        stage = "dataset_loading"
        adapter_kwargs = {"streaming": True} if config.dataset.streaming else {}
        adapter = get_dataset_adapter(config.dataset.name, **adapter_kwargs)
        examples = adapter.load(split=config.dataset.split, limit=config.dataset.limit)
        rank_examples = shard_sequence(examples, rank=state.rank, world_size=state.world_size)

        stage = "prompt_rendering"
        rendered_prompts = render_prompts(rank_examples, config.prompt.template_id)
        logger.info(f"Loaded {len(examples)} examples ({len(rank_examples)} on this rank)")

        stage = "generation"
        completions = build_completions(
            examples=rank_examples,
            prompts=rendered_prompts,
            model=config.model,
            generation=config.generation,
            device=device,
            state=state,
        )

        stage = "verification"
        verifier = build_verifier(config.verifier)
        records = [
            PredictionRecord(
                example=example,
                rendered_prompt=prompt,
                completion=completion,
                verification=verifier.verify(example, completion),
            )
            for example, prompt, completion in zip(
                rank_examples,
                rendered_prompts,
                completions,
                strict=True,
            )
        ]
        logger.info(f"Verified {len(records)} completions with {config.verifier.name}")

        stage = "rank_prediction_write"
        rank_output_dir = ensure_dir(run_dir / "rank_outputs")
        rank_prediction_path = rank_output_dir / f"rank_{state.rank:03d}_predictions.jsonl"
        write_predictions(records, rank_prediction_path)
        barrier(state)

        if state.is_main:
            stage = "prediction_merge"
            rank_files = [
                rank_output_dir / f"rank_{rank:03d}_predictions.jsonl"
                for rank in range(state.world_size)
            ]
            predictions_path = run_dir / "predictions.jsonl"
            merge_jsonl_files(rank_files, predictions_path, sort_key=prediction_row_order_key)
            rows = read_prediction_rows(predictions_path)
            elapsed = perf_counter() - start_time

            stage = "metrics_and_samples"
            metrics = compute_metrics(
                rows,
                extra={
                    "wall_clock_seconds": elapsed,
                    "examples_per_second": len(rows) / elapsed if elapsed else 0.0,
                    "peak_gpu_memory_mb": peak_gpu_memory_mb(),
                    "world_size": state.world_size,
                },
            )
            write_json(run_dir / "metrics.json", metrics)
            write_samples_markdown(rows, run_dir / "samples.md")
            write_status(
                run_dir,
                status="completed",
                stage=stage,
                extra={"wall_clock_seconds": elapsed, "num_predictions": len(rows)},
            )
            logger.info(
                f"Run complete: {run_dir} | {len(rows)} predictions in {elapsed:.2f}s -> metrics.json, predictions.jsonl, samples.md",
            )
        barrier(state)
    except Exception as exc:
        logger.error(f"Run failed at stage {stage!r}: {type(exc).__name__}: {exc}")
        if state.is_main:
            write_status(
                run_dir,
                status="failed",
                stage=stage,
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
        raise
    finally:
        shutdown_distributed(state)
    return run_dir


def validate_config_references(config: EvalConfig) -> None:
    """Fail before run-directory creation when a config references unknown ids."""
    problems = []
    if config.dataset.name.strip().lower() not in DATASET_REGISTRY.names():
        problems.append(
            f"dataset.name={config.dataset.name!r} "
            f"(known: {', '.join(DATASET_REGISTRY.names()) or '<none>'})"
        )

    if config.prompt.template_id not in TEMPLATE_REGISTRY.names():
        problems.append(
            f"prompt.template_id={config.prompt.template_id!r} "
            f"(known: {', '.join(TEMPLATE_REGISTRY.names()) or '<none>'})"
        )

    if config.verifier.name not in VERIFIER_REGISTRY.names():
        problems.append(
            f"verifier.name={config.verifier.name!r} "
            f"(known: {', '.join(VERIFIER_REGISTRY.names()) or '<none>'})"
        )

    if problems:
        msg = "Unknown config reference(s): " + "; ".join(problems)
        raise ValueError(msg)


def build_completions(
    *,
    examples,
    prompts,
    model: ModelConfig,
    generation: GenerationConfig,
    device: str,
    state: DistributedState,
):
    """Generate completions for this rank's prompts via the configured backend.

    Uses the deterministic mock backend when the effective backend is ``"mock"``;
    otherwise loads the model/tokenizer and runs real generation. A bare
    ``"cuda"`` device is resolved to this rank's specific GPU.
    """
    backend = model.backend or generation.backend or "transformers"
    if backend == "mock":
        logger.info(f"Generating {len(prompts)} mock completions (mode={model.mock_mode})")
        return generate_mock_completions(
            examples=examples,
            prompts=prompts,
            mode=model.mock_mode,
        )

    model_name = str(model.name_or_path)
    if device == "cuda":
        device = state.device
    loaded_model, tokenizer = load_causal_lm(
        name_or_path=model_name,
        dtype=model.dtype,
        device=device,
        trust_remote_code=model.trust_remote_code,
    )
    return generate_completions(
        model=loaded_model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=generation.model_dump(),
        model_name_or_path=model_name,
        device=device,
    )


def create_run_dir(config: EvalConfig, config_path: Path, state: DistributedState) -> Path:
    """Create the run directory on rank 0 and broadcast its path to all ranks.

    Honors an explicit ``run.output_dir`` or otherwise builds a timestamped
    directory under ``run.output_root``. Rank 0 owns the timestamped name so all
    ranks agree on a single directory.
    """
    run = config.run
    if state.is_main:
        if run.output_dir:
            run_dir = resolve_project_path(run.output_dir)
        else:
            output_root = resolve_project_path(run.output_root)
            run_name = run.name or config_path.stem
            run_dir = output_root / make_run_id(run_name)
        ensure_dir(run_dir)
        value = str(run_dir)
    else:
        value = None
    # Rank 0 owns the timestamped path, then broadcasts it to avoid divergent run dirs.
    return Path(broadcast_object(value, state))


def collect_env(state: DistributedState) -> dict[str, Any]:
    """Gather reproducibility metadata (versions, git commit, host, ranks) for ``env.json``."""
    packages = {}
    for package in ("torch", "transformers", "datasets", "numpy", "pyyaml"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    root = resolve_project_path(".")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": packages,
        "git_commit": git_commit(root),
        "distributed": {
            "enabled": state.enabled,
            "rank": state.rank,
            "local_rank": state.local_rank,
            "world_size": state.world_size,
            "device": state.device,
        },
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write ``data`` as pretty-printed JSON, creating parent dirs as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=4, ensure_ascii=True), encoding="utf-8")


def write_status(
    run_dir: str | Path,
    *,
    status: str,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a compact run lifecycle marker to ``status.json``."""
    payload = {"status": status, "stage": stage}
    if extra:
        payload.update(extra)
    write_json(Path(run_dir) / "status.json", payload)


def prediction_row_order_key(row: dict[str, Any]) -> tuple[int, int | str]:
    """Sort merged prediction rows back to dataset order when ``row_index`` exists."""
    metadata = row.get("example_metadata")
    if isinstance(metadata, dict):
        row_index = metadata.get("row_index")
        if isinstance(row_index, int):
            return (0, row_index)
        if isinstance(row_index, str) and row_index.isdigit():
            return (0, int(row_index))
    return (1, str(row.get("uid") or ""))
