from __future__ import annotations

from dataclasses import asdict
from typing import Any

from adtool.examples.analysis_metrics.coverage_comparison import compare_discovery_sets
from adtool.examples.analysis_metrics.random_run import run_random_baseline
from fastapi import HTTPException
from adtool.examples.visu.coverage_runs import coverage_runs_dir
from adtool.examples.visu.runtime import (
    DEFAULT_RANDOM_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    RuntimeState,
    ServerConfig,
)
from adtool.examples.visu.server_support import (
    error_detail,
    optional_payload_int,
    payload_int,
    require_directory,
    require_file,
    resolve_input_path,
    timestamped_analysis_dir,
)


def random_run_payload(
    config: ServerConfig,
    state: RuntimeState,
    payload: dict[str, Any],
) -> dict[str, Any]:
    config_file = resolve_input_path(payload.get("config_file"), "config_file")
    if config_file is None:
        raise HTTPException(status_code=422, detail="config_file is required.")
    require_file(config_file, "config_file")

    output_dir = resolve_input_path(payload.get("output_dir"), "output_dir", required=False)
    if output_dir is None:
        output_dir = timestamped_analysis_dir(config.discoveries, "random_run")

    nb_iterations = payload_int(
        payload,
        "nb_iterations",
        DEFAULT_RANDOM_ITERATIONS,
        minimum=1,
    )
    seed = payload_int(
        payload,
        "seed",
        DEFAULT_RANDOM_SEED,
        minimum=0,
        maximum=2**32 - 1,
    )

    with state.analysis_lock:
        try:
            summary = run_random_baseline(
                config_file=config_file,
                output_dir=output_dir,
                nb_iterations=nb_iterations,
                seed=seed,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=error_detail("Random run failed", exc),
            ) from exc

    return {
        "status": "ok",
        "output_dir": str(summary.output_dir),
        "discoveries_dir": str(summary.discoveries_dir),
        "count": summary.count,
        "seed": summary.seed,
    }


def coverage_comparison_payload(
    config: ServerConfig,
    state: RuntimeState,
    payload: dict[str, Any],
) -> dict[str, Any]:
    comparison_path = resolve_input_path(
        payload.get("path", payload.get("comparison_path")),
        "path",
    )
    if comparison_path is None:
        raise HTTPException(status_code=422, detail="path is required.")
    require_directory(comparison_path, "path")

    raw_config_file = payload.get("config_file")
    if isinstance(raw_config_file, str) and raw_config_file.strip().lower() == "none":
        raw_config_file = None
    config_file = resolve_input_path(raw_config_file, "config_file", required=False)
    if config_file is not None:
        require_file(config_file, "config_file")

    points = optional_payload_int(payload, "points")
    label_a = payload.get("label_a") or "IMGEP"
    label_b = payload.get("label_b") or "baseline"

    with state.analysis_lock:
        try:
            summary = compare_discovery_sets(
                config.discoveries,
                comparison_path,
                output_dir=coverage_runs_dir(config),
                label_a=str(label_a),
                label_b=str(label_b),
                config_file=config_file,
                points=points,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=error_detail("Coverage comparison failed", exc),
            ) from exc

    return {
        "status": "ok",
        "run_dir": str(summary.run_dir),
        "dataset_a_path": str(summary.discovery_a_path),
        "dataset_b_path": str(summary.discovery_b_path),
        "dataset_a_count": summary.count_a,
        "dataset_b_count": summary.count_b,
        "dim_count": summary.dim_count,
        "images": [asdict(image) for image in summary.images],
    }
