import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PlotConfig:
    points: int = 256
    bandwidth: Optional[float] = None
    random_color: str = "#4c78a8"
    tool_color: str = "#f58518"
    alpha: float = 0.35
    line_width: float = 2.0
    figsize: Tuple[float, float] = (7.0, 4.0)
    output_format: str = "png"


@dataclass
class CoverageConfig:
    experiment_config_path: Path
    discovery_path: Path
    output_dir: Path
    random_runs: int = 100
    seed: Optional[int] = None
    dimensions: List[int] = None
    embedding_builder: Optional[Dict[str, Any]] = None
    plot: PlotConfig = field(default_factory=PlotConfig)


@dataclass
class CoverageRunSummary:
    run_dir: Path
    random_count: int
    tool_count: int
    dim_count: int
    labels: List[str]
    bounds: List[Tuple[float, float]]
    images: List[str]


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_plot_config(payload: Dict[str, Any]) -> PlotConfig:
    plot_payload = payload.get("plot") or {}
    figsize = plot_payload.get("figsize", None)
    if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
        fig_size = (float(figsize[0]), float(figsize[1]))
    else:
        fig_size = (7.0, 4.0)

    bandwidth = plot_payload.get("bandwidth", None)
    if bandwidth is not None:
        bandwidth = float(bandwidth)

    return PlotConfig(
        points=int(plot_payload.get("points", 256)),
        bandwidth=bandwidth,
        random_color=str(plot_payload.get("random_color", "#4c78a8")),
        tool_color=str(plot_payload.get("tool_color", "#f58518")),
        alpha=float(plot_payload.get("alpha", 0.35)),
        line_width=float(plot_payload.get("line_width", 2.0)),
        figsize=fig_size,
        output_format=str(plot_payload.get("format", "png")),
    )


def load_coverage_config(config_path: Path) -> CoverageConfig:
    with open(config_path, "r") as handle:
        payload = json.load(handle)

    base_dir = config_path.parent
    experiment_config_path = _resolve_path(base_dir, payload["experiment_config_path"])
    discovery_path = _resolve_path(base_dir, payload["discovery_path"])
    output_dir = _resolve_path(base_dir, payload["output_dir"])

    if "dimensions" not in payload:
        raise ValueError("coverage config must include a 'dimensions' list")

    raw_dimensions = payload["dimensions"]
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise ValueError("dimensions must be a non-empty list of integers")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in raw_dimensions):
        raise ValueError("dimensions must contain only integer indices")
    dimensions = list(raw_dimensions)

    return CoverageConfig(
        experiment_config_path=experiment_config_path,
        discovery_path=discovery_path,
        output_dir=output_dir,
        random_runs=int(payload.get("random_runs", 100)),
        seed=payload.get("seed"),
        dimensions=dimensions,
        embedding_builder=payload.get("embedding_builder"),
        plot=_parse_plot_config(payload),
    )
