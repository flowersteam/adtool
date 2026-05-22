import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .config import CoverageRunSummary, load_coverage_config
from .embedding_builder import build_embedding_builder
from .embeddings import collect_random_embeddings, load_discovery_embeddings
from .experiment import build_system_and_explorer, load_experiment_config


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def _align_embeddings(embeddings: List[np.ndarray], dim_count: int) -> List[np.ndarray]:
    return [emb for emb in embeddings if emb.size == dim_count]


def _select_dimensions(configured_dimensions: Any, dim_count: int) -> List[int]:
    if configured_dimensions is None:
        return list(range(dim_count))
    if isinstance(configured_dimensions, str):
        if configured_dimensions.lower() == "all":
            return list(range(dim_count))
        raise ValueError("dimensions must be 'all' or a non-empty list of integers")

    dimensions: List[int] = []
    for idx in configured_dimensions:
        resolved = idx + dim_count if idx < 0 else idx
        if resolved < 0 or resolved >= dim_count:
            raise ValueError(
                f"dimension index {idx} out of range for dim_count {dim_count}"
            )
        dimensions.append(resolved)
    return dimensions


def _compute_bounds_for_dim(
    random_embeddings: List[np.ndarray],
    tool_embeddings: List[np.ndarray],
    dim_index: int,
) -> Tuple[float, float]:
    combined = random_embeddings + tool_embeddings
    values = [emb[dim_index] for emb in combined if emb.size > dim_index]
    if not values:
        min_val, max_val = 0.0, 1.0
    else:
        min_val = float(np.min(values))
        max_val = float(np.max(values))
    if min_val == max_val:
        min_val -= 1.0
        max_val += 1.0
    return min_val, max_val


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def run_coverage_comparison(config_path: Path) -> CoverageRunSummary:
    coverage_config = load_coverage_config(config_path)
    experiment_config = load_experiment_config(coverage_config.experiment_config_path)

    _set_seed(coverage_config.seed)

    system, explorer = build_system_and_explorer(experiment_config)

    build_embedding = build_embedding_builder(coverage_config.embedding_builder)

    random_embeddings = collect_random_embeddings(
        system, explorer, coverage_config.random_runs, build_embedding
    )
    tool_embeddings = load_discovery_embeddings(
        coverage_config.discovery_path, build_embedding
    )

    if not random_embeddings and not tool_embeddings:
        raise RuntimeError("No embeddings collected for coverage comparison.")

    dim_count = len(random_embeddings[0]) if random_embeddings else len(tool_embeddings[0])
    random_embeddings = _align_embeddings(random_embeddings, dim_count)
    tool_embeddings = _align_embeddings(tool_embeddings, dim_count)

    if not random_embeddings and not tool_embeddings:
        raise RuntimeError("No embeddings matched the expected dimension count.")

    configured_dimensions = coverage_config.dimensions
    dimensions = _select_dimensions(configured_dimensions, dim_count)
    labels = [f"dim_{idx}" for idx in dimensions]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = coverage_config.output_dir / f"coverage_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    random_array = np.array(random_embeddings) if random_embeddings else np.empty((0, dim_count))
    tool_array = np.array(tool_embeddings) if tool_embeddings else np.empty((0, dim_count))

    images: List[str] = []

    bounds: List[Tuple[float, float]] = []
    from .plotting import compute_density_curve, plot_density_curves

    for dim, label in zip(dimensions, labels):
        dim_bounds = _compute_bounds_for_dim(random_embeddings, tool_embeddings, dim)
        bounds.append(dim_bounds)

        random_vals = random_array[:, dim] if random_array.size else np.array([])
        tool_vals = tool_array[:, dim] if tool_array.size else np.array([])

        random_curve = compute_density_curve(
            random_vals,
            dim_bounds,
            coverage_config.plot.points,
            coverage_config.plot.bandwidth,
        )
        tool_curve = compute_density_curve(
            tool_vals,
            dim_bounds,
            coverage_config.plot.points,
            coverage_config.plot.bandwidth,
        )

        image_name = f"dim_{dim}_density.{coverage_config.plot.output_format}"
        images.append(image_name)
        plot_density_curves(
            run_dir / image_name,
            random_curve,
            tool_curve,
            label,
            coverage_config.plot.random_color,
            coverage_config.plot.tool_color,
            coverage_config.plot.alpha,
            coverage_config.plot.line_width,
            coverage_config.plot.figsize,
        )

    summary = CoverageRunSummary(
        run_dir=run_dir,
        random_count=len(random_embeddings),
        tool_count=len(tool_embeddings),
        dim_count=dim_count,
        labels=labels,
        bounds=bounds,
        images=images,
    )

    _write_json(
        run_dir / "summary.json",
        {
            "random_count": summary.random_count,
            "tool_count": summary.tool_count,
            "dim_count": summary.dim_count,
            "labels": summary.labels,
            "bounds": [list(b) for b in summary.bounds],
            "images": summary.images,
            "dimensions": configured_dimensions,
            "resolved_dimensions": dimensions,
            "experiment_config": str(coverage_config.experiment_config_path),
            "discovery_path": str(coverage_config.discovery_path),
            "output_dir": str(coverage_config.output_dir),
            "random_baseline": "independent parameter_map.sample() trials",
            "plot_type": "1d marginal KDE density",
            "timestamp": timestamp,
            "plot": asdict(coverage_config.plot),
        },
    )

    return summary
