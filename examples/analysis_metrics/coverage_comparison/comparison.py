from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from pydoc import ErrorDuringImport, locate
from typing import Any, Callable, Optional, Union

import numpy as np

from ..import_paths import ensure_adtool_examples_alias
from .plotting import density_curve, plot_density_curves, plot_dimension_pair_scatter


DEFAULT_OUTPUT_DIR = Path("coverage_runs")
DEFAULT_POINTS = 256
DEFAULT_COLOR_A = "#4c78a8"
DEFAULT_COLOR_B = "#f58518"
DEFAULT_ALPHA = 0.35
DEFAULT_LINE_WIDTH = 2.0
DEFAULT_FIGSIZE = (7.0, 4.0)
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_DIMENSIONS = "all"


@dataclass(frozen=True)
class DiscoverySet:
    path: Path
    files: list[Path]
    payloads: list[dict[str, Any]]
    outputs: np.ndarray


PretreatmentResult = Union[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, list[str]],
]
DimensionPretreatmentFn = Callable[
    [DiscoverySet, DiscoverySet, dict[str, Any]],
    PretreatmentResult,
]


@dataclass(frozen=True)
class PlotConfig:
    points: int = DEFAULT_POINTS
    color_a: str = DEFAULT_COLOR_A
    color_b: str = DEFAULT_COLOR_B
    alpha: float = DEFAULT_ALPHA
    line_width: float = DEFAULT_LINE_WIDTH
    figsize: tuple[float, float] = DEFAULT_FIGSIZE
    output_format: str = DEFAULT_OUTPUT_FORMAT


@dataclass(frozen=True)
class DimensionPretreatmentConfig:
    path: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageImageSummary:
    file: str
    title: str
    plot_type: str
    dimensions: list[int]
    bounds: list[tuple[float, float]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "title": self.title,
            "plot_type": self.plot_type,
            "dimensions": self.dimensions,
            "bounds": [list(bounds) for bounds in self.bounds],
        }


@dataclass(frozen=True)
class ComparisonConfig:
    dimensions: Union[str, list[int]] = DEFAULT_DIMENSIONS
    additional_2d_graphs: list[tuple[int, int]] = field(default_factory=list)
    dimension_pretreatment: Optional[DimensionPretreatmentConfig] = None
    plot: PlotConfig = field(default_factory=PlotConfig)


@dataclass(frozen=True)
class CoverageComparisonSummary:
    run_dir: Path
    discovery_a_path: Path
    discovery_b_path: Path
    label_a: str
    label_b: str
    count_a: int
    count_b: int
    dim_count: int
    labels: list[str]
    resolved_dimensions: list[int]
    resolved_additional_2d_graphs: list[tuple[int, int]]
    dimension_pretreatment: Optional[str]
    bounds: list[tuple[float, float]]
    images: list[CoverageImageSummary]


def _parse_dimensions(value: Any) -> Union[str, list[int]]:
    if value is None or value == DEFAULT_DIMENSIONS:
        return DEFAULT_DIMENSIONS
    if isinstance(value, list):
        return [int(dim) for dim in value]
    raise ValueError("dimensions must be 'all' or a list of integers")


def _parse_additional_2d_graphs(value: Any) -> list[tuple[int, int]]:
    if not value:
        return []
    return [(int(x_dim), int(y_dim)) for x_dim, y_dim in value]


def _parse_plot_config(value: Any) -> PlotConfig:
    value = value or {}
    return PlotConfig(
        points=max(2, int(value.get("points", DEFAULT_POINTS))),
        color_a=str(value.get("color_a", value.get("dataset_a_color", DEFAULT_COLOR_A))),
        color_b=str(value.get("color_b", value.get("dataset_b_color", DEFAULT_COLOR_B))),
        alpha=float(value.get("alpha", DEFAULT_ALPHA)),
        line_width=float(value.get("line_width", DEFAULT_LINE_WIDTH)),
        figsize=tuple(map(float, value.get("figsize", DEFAULT_FIGSIZE))),
        output_format=str(value.get("format", DEFAULT_OUTPUT_FORMAT)).lstrip("."),
    )


def _parse_dimension_pretreatment(value: Any) -> Optional[DimensionPretreatmentConfig]:
    if not value:
        return None
    if isinstance(value, str):
        return DimensionPretreatmentConfig(path=value)
    return DimensionPretreatmentConfig(
        path=str(value["path"]),
        config=dict(value.get("config") or {}),
    )


def load_comparison_config(config_path: Optional[Union[str, Path]]) -> ComparisonConfig:
    if config_path is None:
        return ComparisonConfig()

    with Path(config_path).open("r") as handle:
        payload = json.load(handle)

    return ComparisonConfig(
        dimensions=_parse_dimensions(payload.get("dimensions")),
        additional_2d_graphs=_parse_additional_2d_graphs(
            payload.get("additional_2d_graphs")
        ),
        dimension_pretreatment=_parse_dimension_pretreatment(
            payload.get("dimension_pretreatment", payload.get("pretreatment"))
        ),
        plot=_parse_plot_config(payload.get("plot")),
    )


def _locate_dotted_callable(path: str) -> DimensionPretreatmentFn:
    ensure_adtool_examples_alias()
    try:
        fn = locate(path)
    except ErrorDuringImport:
        raise

    if fn is None and path.startswith("adtool.examples."):
        fn = locate(f"examples.{path[len('adtool.examples.'):]}")
    if fn is None or not callable(fn):
        raise ValueError(f"Could not retrieve dimension pretreatment from path: {path}")
    return fn


def _load_dimension_pretreatment(
    config: Optional[DimensionPretreatmentConfig],
) -> Optional[DimensionPretreatmentFn]:
    return None if config is None else _locate_dotted_callable(config.path)


def _load_discovery_output(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with path.open("r") as handle:
        payload = json.load(handle)
    output = np.asarray(payload["output"], dtype=float).reshape(-1)
    if output.size == 0:
        raise ValueError(f"Discovery output is empty: {path}")
    return payload, output


def load_discovery_set(discovery_path: Path) -> DiscoverySet:
    discovery_path = Path(discovery_path)
    files = sorted(
        (
            path
            for path in discovery_path.rglob("discovery.json")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        raise ValueError(f"No discovery.json files found under: {discovery_path}")

    loaded = [_load_discovery_output(path) for path in files]
    payloads = [payload for payload, _ in loaded]
    outputs = np.vstack([output for _, output in loaded])
    if len({output.shape[0] for _, output in loaded}) != 1:
        raise ValueError(f"Discovery output dimension mismatch under: {discovery_path}")

    return DiscoverySet(
        path=discovery_path,
        files=files,
        payloads=payloads,
        outputs=outputs,
    )


def load_discovery_outputs(discovery_path: Path) -> np.ndarray:
    return load_discovery_set(discovery_path).outputs


def _apply_dimension_pretreatment(
    dataset_a: DiscoverySet,
    dataset_b: DiscoverySet,
    config: Optional[DimensionPretreatmentConfig],
) -> tuple[np.ndarray, np.ndarray, Optional[list[str]]]:
    if config is None:
        return dataset_a.outputs, dataset_b.outputs, None

    result = _load_dimension_pretreatment(config)(dataset_a, dataset_b, config.config)
    values_a = np.asarray(result[0], dtype=float)
    values_b = np.asarray(result[1], dtype=float)
    labels = None if len(result) < 3 else list(result[2])
    return values_a, values_b, labels


def _resolve_index(index: int, dim_count: int) -> int:
    index = dim_count + index if index < 0 else index
    if index < 0 or index >= dim_count:
        raise ValueError(f"dimension index {index} out of range for dim_count {dim_count}")
    return index


def _resolve_dimensions(
    dimensions: Union[str, list[int]],
    dim_count: int,
) -> list[int]:
    if dimensions == DEFAULT_DIMENSIONS:
        return list(range(dim_count))
    return [_resolve_index(dim, dim_count) for dim in dimensions]


def _resolve_dimension_pairs(
    dimension_pairs: list[tuple[int, int]],
    dim_count: int,
) -> list[tuple[int, int]]:
    return [
        (_resolve_index(x_dim, dim_count), _resolve_index(y_dim, dim_count))
        for x_dim, y_dim in dimension_pairs
    ]


def _display_dimension_label(label: str, dim_index: int) -> str:
    return f"{label} ({dim_index})"


def _default_label(path: Path) -> str:
    return path.name or str(path)


def _value_min_max(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    return (
        min(float(np.min(values_a)), float(np.min(values_b))),
        max(float(np.max(values_a)), float(np.max(values_b))),
    )


def _value_bounds(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    min_value, max_value = _value_min_max(values_a, values_b)
    if min_value == max_value:
        return min_value - 1.0, max_value + 1.0
    return min_value, max_value


def _is_integer_values(values_a: np.ndarray, values_b: np.ndarray) -> bool:
    return bool(
        np.allclose(values_a, np.round(values_a))
        and np.allclose(values_b, np.round(values_b))
    )


def _plot_bounds(values_a: np.ndarray, values_b: np.ndarray) -> tuple[bool, tuple[float, float]]:
    integer_values = _is_integer_values(values_a, values_b)
    min_value, max_value = _value_min_max(values_a, values_b)
    if integer_values:
        return True, (min_value - 0.5, max_value + 0.5)
    if min_value == max_value:
        return False, (min_value - 1.0, max_value + 1.0)
    return False, (min_value, max_value)


def _run_dir(output_dir: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"coverage_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, timestamp


def _summary_payload(
    summary: CoverageComparisonSummary,
    timestamp: str,
    config: ComparisonConfig,
) -> dict[str, Any]:
    return {
        "run_dir": str(summary.run_dir),
        "comparison_type": "discovery_vs_discovery",
        "discovery_a_path": str(summary.discovery_a_path),
        "discovery_b_path": str(summary.discovery_b_path),
        "dataset_a_label": summary.label_a,
        "dataset_b_label": summary.label_b,
        "dataset_a_count": summary.count_a,
        "dataset_b_count": summary.count_b,
        "dim_count": summary.dim_count,
        "labels": summary.labels,
        "resolved_dimensions": summary.resolved_dimensions,
        "resolved_additional_2d_graphs": [
            list(dim_pair) for dim_pair in summary.resolved_additional_2d_graphs
        ],
        "dimension_pretreatment": summary.dimension_pretreatment,
        "dimension_pretreatment_config": (
            {} if config.dimension_pretreatment is None else config.dimension_pretreatment.config
        ),
        "bounds": [list(bounds) for bounds in summary.bounds],
        "images": [image.to_payload() for image in summary.images],
        "plot_type": "coverage comparison plots",
        "timestamp": timestamp,
        "plot": {
            "points": config.plot.points,
            "dataset_a_color": config.plot.color_a,
            "dataset_b_color": config.plot.color_b,
            "alpha": config.plot.alpha,
            "line_width": config.plot.line_width,
            "figsize": list(config.plot.figsize),
            "format": config.plot.output_format,
        },
    }


def compare_discovery_sets(
    discovery_a_path: Union[str, Path],
    discovery_b_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    config_file: Optional[Union[str, Path]] = None,
    points: Optional[int] = None,
) -> CoverageComparisonSummary:
    config = load_comparison_config(config_file)
    if points is not None:
        config = replace(config, plot=replace(config.plot, points=max(2, int(points))))

    discovery_a_path = Path(discovery_a_path).resolve()
    discovery_b_path = Path(discovery_b_path).resolve()
    label_a = label_a or _default_label(discovery_a_path)
    label_b = label_b or _default_label(discovery_b_path)

    dataset_a = load_discovery_set(discovery_a_path)
    dataset_b = load_discovery_set(discovery_b_path)
    values_a, values_b, custom_labels = _apply_dimension_pretreatment(
        dataset_a,
        dataset_b,
        config.dimension_pretreatment,
    )
    dim_count = values_a.shape[1]
    if values_b.shape[1] != dim_count:
        raise ValueError("Discovery sets must have the same output dimension count")

    raw_labels = custom_labels or [f"dim_{idx}" for idx in range(dim_count)]
    dimensions = _resolve_dimensions(config.dimensions, dim_count)
    additional_2d_graphs = _resolve_dimension_pairs(
        config.additional_2d_graphs,
        dim_count,
    )
    labels = [_display_dimension_label(raw_labels[idx], idx) for idx in dimensions]

    run_dir, timestamp = _run_dir(Path(output_dir).resolve())
    bounds: list[tuple[float, float]] = []
    images: list[CoverageImageSummary] = []

    for dim_index, dim_label in zip(dimensions, labels):
        dim_values_a = values_a[:, dim_index]
        dim_values_b = values_b[:, dim_index]
        integer_values, dim_bounds = _plot_bounds(dim_values_a, dim_values_b)
        image_name = f"dim_{dim_index}_density.{config.plot.output_format}"
        bounds.append(dim_bounds)
        images.append(
            CoverageImageSummary(
                file=image_name,
                title=dim_label,
                plot_type="1d density",
                dimensions=[dim_index],
                bounds=[dim_bounds],
            )
        )
        plot_density_curves(
            run_dir / image_name,
            density_curve(
                dim_values_a,
                dim_bounds,
                config.plot.points,
                integer_values=integer_values,
            ),
            density_curve(
                dim_values_b,
                dim_bounds,
                config.plot.points,
                integer_values=integer_values,
            ),
            dim_label,
            label_a,
            label_b,
            config.plot,
            integer_x=integer_values,
        )

    for x_dim, y_dim in additional_2d_graphs:
        x_values_a = values_a[:, x_dim]
        x_values_b = values_b[:, x_dim]
        y_values_a = values_a[:, y_dim]
        y_values_b = values_b[:, y_dim]
        x_label = _display_dimension_label(raw_labels[x_dim], x_dim)
        y_label = _display_dimension_label(raw_labels[y_dim], y_dim)
        image_name = f"dims_{x_dim}_{y_dim}_scatter.{config.plot.output_format}"
        images.append(
            CoverageImageSummary(
                file=image_name,
                title=f"X = {x_label} | Y = {y_label}",
                plot_type="2d scatter",
                dimensions=[x_dim, y_dim],
                bounds=[
                    _value_bounds(x_values_a, x_values_b),
                    _value_bounds(y_values_a, y_values_b),
                ],
            )
        )
        plot_dimension_pair_scatter(
            run_dir / image_name,
            x_values_a,
            y_values_a,
            x_values_b,
            y_values_b,
            x_label,
            y_label,
            label_a,
            label_b,
            config.plot,
        )

    summary = CoverageComparisonSummary(
        run_dir=run_dir,
        discovery_a_path=discovery_a_path,
        discovery_b_path=discovery_b_path,
        label_a=label_a,
        label_b=label_b,
        count_a=values_a.shape[0],
        count_b=values_b.shape[0],
        dim_count=dim_count,
        labels=labels,
        resolved_dimensions=dimensions,
        resolved_additional_2d_graphs=additional_2d_graphs,
        dimension_pretreatment=(
            None if config.dimension_pretreatment is None else config.dimension_pretreatment.path
        ),
        bounds=bounds,
        images=images,
    )

    with (run_dir / "summary.json").open("w") as handle:
        json.dump(_summary_payload(summary, timestamp, config), handle, indent=2)

    return summary


run_coverage_comparison = compare_discovery_sets
