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
DEFAULT_ADDITIONAL_2D_GRAPHS: list[tuple[int, int]] = []


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
    additional_2d_graphs: list[tuple[int, int]] = field(
        default_factory=lambda: list(DEFAULT_ADDITIONAL_2D_GRAPHS)
    )
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


@dataclass(frozen=True)
class DimensionPlotAnalysis:
    integer_values: bool
    bounds: tuple[float, float]


@dataclass(frozen=True)
class PreparedComparison:
    values_a: np.ndarray
    values_b: np.ndarray
    dim_count: int
    raw_labels: list[str]
    resolved_dimensions: list[int]
    resolved_additional_2d_graphs: list[tuple[int, int]]


def _parse_dimensions(value: Any) -> Union[str, list[int]]:
    if value is None:
        return DEFAULT_DIMENSIONS

    if isinstance(value, str):
        if value.lower() == DEFAULT_DIMENSIONS:
            return DEFAULT_DIMENSIONS
        raise ValueError("dimensions must be 'all' or a list of integers")

    if isinstance(value, list):
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            raise ValueError("dimensions must contain only integer indices")
        return list(value)

    raise ValueError("dimensions must be 'all' or a list of integers")


def _parse_figsize(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return DEFAULT_FIGSIZE


def _parse_additional_2d_graphs(value: Any) -> list[tuple[int, int]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("additional_2d_graphs must be a list of [x_dim, y_dim] pairs")

    resolved_pairs: list[tuple[int, int]] = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "additional_2d_graphs must contain only [x_dim, y_dim] pairs"
            )
        x_dim, y_dim = pair
        if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in (x_dim, y_dim)):
            raise ValueError(
                "additional_2d_graphs pairs must contain only integer indices"
            )
        resolved_pairs.append((x_dim, y_dim))
    return resolved_pairs


def _parse_output_format(value: Any) -> str:
    output_format = str(value or DEFAULT_OUTPUT_FORMAT).lstrip(".")
    if not output_format:
        raise ValueError("plot format cannot be empty")
    return output_format


def _parse_plot_config(value: Any) -> PlotConfig:
    if value is None:
        return PlotConfig()
    if not isinstance(value, dict):
        raise ValueError("plot must be a JSON object when provided")

    return PlotConfig(
        points=max(2, int(value.get("points", DEFAULT_POINTS))),
        color_a=str(value.get("color_a", value.get("dataset_a_color", DEFAULT_COLOR_A))),
        color_b=str(value.get("color_b", value.get("dataset_b_color", DEFAULT_COLOR_B))),
        alpha=float(value.get("alpha", DEFAULT_ALPHA)),
        line_width=float(value.get("line_width", DEFAULT_LINE_WIDTH)),
        figsize=_parse_figsize(value.get("figsize")),
        output_format=_parse_output_format(value.get("format")),
    )


def _parse_dimension_pretreatment(value: Any) -> Optional[DimensionPretreatmentConfig]:
    if value is None or value == "":
        return None

    if isinstance(value, str):
        return DimensionPretreatmentConfig(path=value)

    if isinstance(value, dict):
        path = value.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("dimension_pretreatment.path must be a dotted callable path")

        pretreatment_config = value.get("config", {})
        if pretreatment_config is None:
            pretreatment_config = {}
        if not isinstance(pretreatment_config, dict):
            raise ValueError("dimension_pretreatment.config must be a JSON object")

        return DimensionPretreatmentConfig(
            path=path,
            config=dict(pretreatment_config),
        )

    raise ValueError(
        "dimension_pretreatment must be null, a dotted callable path, "
        "or an object with path/config"
    )


def load_comparison_config(config_path: Optional[Union[str, Path]]) -> ComparisonConfig:
    if config_path is None:
        return ComparisonConfig()

    with Path(config_path).open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("coverage comparison config must be a JSON object")

    return ComparisonConfig(
        dimensions=_parse_dimensions(payload.get("dimensions", DEFAULT_DIMENSIONS)),
        additional_2d_graphs=_parse_additional_2d_graphs(
            payload.get("additional_2d_graphs")
        ),
        dimension_pretreatment=_parse_dimension_pretreatment(
            payload.get(
                "dimension_pretreatment",
                payload.get("pretreatment"),
            )
        ),
        plot=_parse_plot_config(payload.get("plot")),
    )


def _locate_dotted_callable(path: str) -> DimensionPretreatmentFn:
    ensure_adtool_examples_alias()
    try:
        candidate = locate(path)
    except ErrorDuringImport:
        raise

    if candidate is None and path.startswith("adtool.examples."):
        candidate = locate(f"examples.{path[len('adtool.examples.'):]}")

    if candidate is None:
        raise ValueError(f"Could not retrieve dimension pretreatment from path: {path}")
    if not callable(candidate):
        raise ValueError(f"Dimension pretreatment path is not callable: {path}")
    return candidate


def _load_dimension_pretreatment(
    config: Optional[DimensionPretreatmentConfig],
) -> Optional[DimensionPretreatmentFn]:
    if config is None:
        return None
    return _locate_dotted_callable(config.path)


def _discovery_files(discovery_path: Path) -> list[Path]:
    files = [path for path in discovery_path.rglob("discovery.json") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime)


def _load_discovery_payload(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or "output" not in payload:
        raise ValueError(f"Discovery is missing a numeric output: {path}")
    return payload


def _extract_discovery_output(payload: dict[str, Any], path: Path) -> np.ndarray:
    try:
        embedding = np.asarray(payload["output"], dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Discovery output is not numeric: {path}") from exc

    if embedding.size == 0:
        raise ValueError(f"Discovery output is empty: {path}")
    if np.isnan(embedding).any() or np.isinf(embedding).any():
        raise ValueError(f"Discovery output contains NaN or infinity: {path}")

    return embedding


def load_discovery_set(discovery_path: Path) -> DiscoverySet:
    discovery_path = Path(discovery_path)
    files = _discovery_files(discovery_path)
    if not files:
        raise ValueError(f"No discovery.json files found under: {discovery_path}")

    payloads = [_load_discovery_payload(path) for path in files]
    embeddings = [
        _extract_discovery_output(payload, path)
        for payload, path in zip(payloads, files)
    ]
    dim_count = embeddings[0].size
    for path, embedding in zip(files, embeddings):
        if embedding.size != dim_count:
            raise ValueError(
                f"Discovery output dimension mismatch in {path}: "
                f"expected {dim_count}, got {embedding.size}"
            )

    return DiscoverySet(
        path=discovery_path,
        files=files,
        payloads=payloads,
        outputs=np.vstack(embeddings),
    )


def load_discovery_outputs(discovery_path: Path) -> np.ndarray:
    return load_discovery_set(discovery_path).outputs


def _validate_matching_dimensions(values_a: np.ndarray, values_b: np.ndarray) -> int:
    dim_count = values_a.shape[1]
    if values_b.shape[1] != dim_count:
        raise ValueError(
            "Discovery sets must have the same output dimension count: "
            f"{dim_count} != {values_b.shape[1]}"
        )
    return dim_count


def _resolve_dimension_index(dim_index: int, dim_count: int) -> int:
    resolved = dim_index + dim_count if dim_index < 0 else dim_index
    if resolved < 0 or resolved >= dim_count:
        raise ValueError(
            f"dimension index {dim_index} out of range for dim_count {dim_count}"
        )
    return resolved


def _select_dimensions(dimensions: Union[str, list[int]], dim_count: int) -> list[int]:
    if dimensions == DEFAULT_DIMENSIONS:
        return list(range(dim_count))

    return [_resolve_dimension_index(dim_index, dim_count) for dim_index in dimensions]


def _select_dimension_pairs(
    dimensions: list[tuple[int, int]],
    dim_count: int,
) -> list[tuple[int, int]]:
    return [
        (
            _resolve_dimension_index(x_dim, dim_count),
            _resolve_dimension_index(y_dim, dim_count),
        )
        for x_dim, y_dim in dimensions
    ]


def _validate_values(values: np.ndarray, name: str) -> None:
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one dimension")
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError(f"{name} contains NaN or infinity")


def _apply_dimension_pretreatment(
    dataset_a: DiscoverySet,
    dataset_b: DiscoverySet,
    pretreatment_fn: Optional[DimensionPretreatmentFn],
    pretreatment_config: Optional[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, Optional[list[str]]]:
    if pretreatment_fn is None:
        return dataset_a.outputs, dataset_b.outputs, None

    result = pretreatment_fn(dataset_a, dataset_b, pretreatment_config or {})
    if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
        raise ValueError(
            "Dimension pretreatment must return values_a, values_b, "
            "and optionally a list of labels"
        )

    treated_a = np.asarray(result[0], dtype=float)
    treated_b = np.asarray(result[1], dtype=float)
    labels = list(result[2]) if len(result) == 3 else None

    _validate_values(treated_a, "Pretreated discovery set A")
    _validate_values(treated_b, "Pretreated discovery set B")
    if (
        treated_a.shape[0] != dataset_a.outputs.shape[0]
        or treated_b.shape[0] != dataset_b.outputs.shape[0]
    ):
        raise ValueError(
            "Dimension pretreatment must preserve discovery counts: "
            f"{treated_a.shape[0]} != {dataset_a.outputs.shape[0]} or "
            f"{treated_b.shape[0]} != {dataset_b.outputs.shape[0]}"
        )
    if treated_a.shape[1] != treated_b.shape[1]:
        raise ValueError(
            "Pretreated discovery sets must have the same dimension count: "
            f"{treated_a.shape[1]} != {treated_b.shape[1]}"
        )
    if labels is not None and len(labels) != treated_a.shape[1]:
        raise ValueError(
            "Dimension pretreatment labels must match the pretreated dimension count"
        )

    return treated_a, treated_b, labels


def _value_min_max(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    min_value = min(float(np.min(values_a)), float(np.min(values_b)))
    max_value = max(float(np.max(values_a)), float(np.max(values_b)))
    return min_value, max_value


def _value_bounds(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    min_value, max_value = _value_min_max(values_a, values_b)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    return min_value, max_value


def _integer_values(values_a: np.ndarray, values_b: np.ndarray) -> bool:
    return bool(
        np.allclose(values_a, np.round(values_a))
        and np.allclose(values_b, np.round(values_b))
    )


def _plot_bounds(
    values_a: np.ndarray,
    values_b: np.ndarray,
    integer_values: bool,
) -> tuple[float, float]:
    if not integer_values:
        return _value_bounds(values_a, values_b)

    min_value, max_value = _value_min_max(values_a, values_b)
    return min_value - 0.5, max_value + 0.5


def _run_dir(output_dir: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"coverage_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, timestamp


def _default_label(path: Path) -> str:
    return path.name or str(path)


def _display_dimension_label(label: str, dim_index: int) -> str:
    return f"{label} ({dim_index})"


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
            config.dimension_pretreatment.config
            if config.dimension_pretreatment is not None
            else {}
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
    prepared = _prepare_comparison(
        discovery_a_path,
        discovery_b_path,
        config,
    )
    run_dir, timestamp = _run_dir(Path(output_dir).resolve())
    labels = _selected_dimension_labels(
        prepared.raw_labels,
        prepared.resolved_dimensions,
    )
    bounds, density_images = _generate_density_plots(
        prepared,
        labels,
        run_dir,
        label_a,
        label_b,
        config.plot,
    )
    scatter_images = _generate_scatter_plots(
        prepared,
        run_dir,
        label_a,
        label_b,
        config.plot,
    )
    images = density_images + scatter_images

    summary = CoverageComparisonSummary(
        run_dir=run_dir,
        discovery_a_path=discovery_a_path,
        discovery_b_path=discovery_b_path,
        label_a=label_a,
        label_b=label_b,
        count_a=prepared.values_a.shape[0],
        count_b=prepared.values_b.shape[0],
        dim_count=prepared.dim_count,
        labels=labels,
        resolved_dimensions=prepared.resolved_dimensions,
        resolved_additional_2d_graphs=prepared.resolved_additional_2d_graphs,
        dimension_pretreatment=(
            config.dimension_pretreatment.path
            if config.dimension_pretreatment is not None
            else None
        ),
        bounds=bounds,
        images=images,
    )

    payload = _summary_payload(summary, timestamp, config)
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)

    return summary


def _prepare_comparison(
    discovery_a_path: Path,
    discovery_b_path: Path,
    config: ComparisonConfig,
) -> PreparedComparison:
    dataset_a = load_discovery_set(discovery_a_path)
    dataset_b = load_discovery_set(discovery_b_path)
    _validate_matching_dimensions(dataset_a.outputs, dataset_b.outputs)

    pretreatment_fn = _load_dimension_pretreatment(config.dimension_pretreatment)
    pretreatment_config = (
        config.dimension_pretreatment.config
        if config.dimension_pretreatment is not None
        else None
    )
    values_a, values_b, pretreated_labels = _apply_dimension_pretreatment(
        dataset_a,
        dataset_b,
        pretreatment_fn,
        pretreatment_config,
    )
    dim_count = _validate_matching_dimensions(values_a, values_b)
    raw_labels = pretreated_labels or [f"dim_{idx}" for idx in range(dim_count)]
    return PreparedComparison(
        values_a=values_a,
        values_b=values_b,
        dim_count=dim_count,
        raw_labels=raw_labels,
        resolved_dimensions=_select_dimensions(config.dimensions, dim_count),
        resolved_additional_2d_graphs=_select_dimension_pairs(
            config.additional_2d_graphs,
            dim_count,
        ),
    )


def _selected_dimension_labels(
    raw_labels: list[str],
    resolved_dimensions: list[int],
) -> list[str]:
    return [_display_dimension_label(raw_labels[idx], idx) for idx in resolved_dimensions]


def _generate_density_plots(
    prepared: PreparedComparison,
    labels: list[str],
    run_dir: Path,
    label_a: str,
    label_b: str,
    plot_config: PlotConfig,
) -> tuple[list[tuple[float, float]], list[CoverageImageSummary]]:
    bounds: list[tuple[float, float]] = []
    images: list[CoverageImageSummary] = []

    for dim_index, dim_label in zip(prepared.resolved_dimensions, labels):
        dim_values_a = prepared.values_a[:, dim_index]
        dim_values_b = prepared.values_b[:, dim_index]
        analysis = _analyze_dimension_plot(dim_values_a, dim_values_b)
        image_name = f"dim_{dim_index}_density.{plot_config.output_format}"
        bounds.append(analysis.bounds)
        images.append(
            CoverageImageSummary(
                file=image_name,
                title=dim_label,
                plot_type="1d density",
                dimensions=[dim_index],
                bounds=[analysis.bounds],
            )
        )
        plot_density_curves(
            run_dir / image_name,
            density_curve(
                dim_values_a,
                analysis.bounds,
                plot_config.points,
                integer_values=analysis.integer_values,
            ),
            density_curve(
                dim_values_b,
                analysis.bounds,
                plot_config.points,
                integer_values=analysis.integer_values,
            ),
            dim_label,
            label_a,
            label_b,
            plot_config,
            integer_x=analysis.integer_values,
        )

    return bounds, images


def _generate_scatter_plots(
    prepared: PreparedComparison,
    run_dir: Path,
    label_a: str,
    label_b: str,
    plot_config: PlotConfig,
) -> list[CoverageImageSummary]:
    images: list[CoverageImageSummary] = []

    for x_dim, y_dim in prepared.resolved_additional_2d_graphs:
        x_label = _display_dimension_label(prepared.raw_labels[x_dim], x_dim)
        y_label = _display_dimension_label(prepared.raw_labels[y_dim], y_dim)
        x_values_a = prepared.values_a[:, x_dim]
        x_values_b = prepared.values_b[:, x_dim]
        y_values_a = prepared.values_a[:, y_dim]
        y_values_b = prepared.values_b[:, y_dim]
        image_name = f"dims_{x_dim}_{y_dim}_scatter.{plot_config.output_format}"
        x_bounds = _value_bounds(x_values_a, x_values_b)
        y_bounds = _value_bounds(y_values_a, y_values_b)
        images.append(
            CoverageImageSummary(
                file=image_name,
                title=f"X = {x_label} | Y = {y_label}",
                plot_type="2d scatter",
                dimensions=[x_dim, y_dim],
                bounds=[x_bounds, y_bounds],
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
            plot_config,
        )

    return images


def _analyze_dimension_plot(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> DimensionPlotAnalysis:
    integer_values = _integer_values(values_a, values_b)
    return DimensionPlotAnalysis(
        integer_values=integer_values,
        bounds=_plot_bounds(values_a, values_b, integer_values),
    )


run_coverage_comparison = compare_discovery_sets
