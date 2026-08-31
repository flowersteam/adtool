from __future__ import annotations

import logging
from typing import Any

from ..analysis_metrics.analysis_run import run_analysis
from ..analysis_metrics.random_run import run_random_baseline
from fastapi import HTTPException
from .analysis_runs import analysis_runs_dir
from .runtime import (
    DEFAULT_RANDOM_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    RuntimeState,
    ServerConfig,
)
from .server_support import (
    error_detail,
    payload_int,
    require_directory,
    require_file,
    resolve_input_path,
)

LOGGER = logging.getLogger("uvicorn.error")


def random_run_payload(
    config: ServerConfig,
    state: RuntimeState,
    payload: dict[str, Any],
) -> dict[str, Any]:
    config_file = resolve_input_path(payload.get("config_file"), "config_file")
    if config_file is None:
        raise HTTPException(status_code=422, detail="config_file is required.")
    require_file(config_file, "config_file")

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
                nb_iterations=nb_iterations,
                seed=seed,
            )
        except Exception as exc:
            LOGGER.exception("Random analysis run failed")
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


def run_analysis_payload(
    config: ServerConfig,
    state: RuntimeState,
    payload: dict[str, Any],
) -> dict[str, Any]:
    raw_paths = payload.get("discovery_paths")
    if not isinstance(raw_paths, list):
        raise HTTPException(status_code=422, detail="discovery_paths must be a list.")

    discovery_paths = []
    for index, raw_path in enumerate(raw_paths):
        resolved = resolve_input_path(raw_path, f"discovery_paths[{index}]")
        if resolved is None:
            raise HTTPException(
                status_code=422,
                detail=f"discovery_paths[{index}] must be a directory path.",
            )
        require_directory(resolved, f"discovery_paths[{index}]")
        discovery_paths.append(resolved)
    if not discovery_paths:
        raise HTTPException(status_code=422, detail="At least one discovery path is required.")

    raw_config_file = payload.get("config_file")
    if isinstance(raw_config_file, str) and raw_config_file.strip().lower() == "none":
        raw_config_file = None
    config_file = resolve_input_path(raw_config_file, "config_file", required=False)
    if config_file is not None:
        require_file(config_file, "config_file")

    raw_labels = payload.get("labels") or []
    if not isinstance(raw_labels, list):
        raise HTTPException(status_code=422, detail="labels must be a list.")

    with state.analysis_lock:
        try:
            summary = run_analysis(
                discovery_paths,
                output_dir=analysis_runs_dir(config),
                labels=[str(label) for label in raw_labels],
                config_file=config_file,
            )
        except Exception as exc:
            LOGGER.exception("Analysis run failed")
            raise HTTPException(
                status_code=500,
                detail=error_detail("Analysis run failed", exc),
            ) from exc

    return {
        "status": "ok",
        "run_dir": str(summary.run_dir),
        "datasets": [dataset.to_payload() for dataset in summary.datasets],
        "module_order": list(summary.module_order),
        "modules": dict(summary.modules),
    }
