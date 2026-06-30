import numpy as np

from .plotting import (
    density_curve,
    plot_density_curves,
)
from ..shared import AnalysisImage, apply_projection


def _resolve_dimensions(dimensions, dim_count):
    if dimensions == "all":
        return list(range(dim_count))
    return [int(dim) for dim in dimensions]


def _display_dimension_label(label, dim_index):
    return f"{label} ({dim_index})"


def _value_bounds(series_values):
    min_value = min(float(np.min(values)) for values in series_values)
    max_value = max(float(np.max(values)) for values in series_values)
    if min_value == max_value:
        return min_value - 1.0, max_value + 1.0
    return min_value, max_value


def _plot_bounds(series_values):
    integer_values = bool(
        all(np.allclose(values, np.round(values)) for values in series_values)
    )
    min_value = min(float(np.min(values)) for values in series_values)
    max_value = max(float(np.max(values)) for values in series_values)
    if integer_values:
        return True, (min_value - 0.5, max_value + 0.5)
    if min_value == max_value:
        return False, (min_value - 1.0, max_value + 1.0)
    return False, (min_value, max_value)


def run_comparison_1d(config, datasets, labels, run_dir):
    projected_values, raw_labels = apply_projection(config.projection, datasets)
    dim_count = projected_values[0].shape[1]
    raw_labels = raw_labels or [f"dim_{idx}" for idx in range(dim_count)]
    dimensions = _resolve_dimensions(config.dimensions, dim_count)

    images = []
    dimension_labels = []
    bounds = []
    for dim_index in dimensions:
        dim_label = _display_dimension_label(raw_labels[dim_index], dim_index)
        dimension_series = [values[:, dim_index] for values in projected_values]
        integer_values, dim_bounds = _plot_bounds(dimension_series)
        image_name = f"comparison_1d_dim_{dim_index}.{config.plot.output_format}"
        plot_density_curves(
            run_dir / image_name,
            [
                density_curve(
                    values,
                    dim_bounds,
                    config.plot.points,
                    integer_values=integer_values,
                )
                for values in dimension_series
            ],
            dim_label,
            labels,
            config.plot,
            integer_x=integer_values,
        )
        images.append(
            AnalysisImage(
                file=image_name,
                title=dim_label,
                plot_type="1d density",
                dimensions=[dim_index],
                bounds=[dim_bounds],
            ).to_payload()
        )
        dimension_labels.append(dim_label)
        bounds.append(list(_value_bounds(dimension_series)))

    return {
        "title": "1D comparison",
        "images": images,
        "dimensions": list(dimensions),
        "labels": dimension_labels,
        "bounds": bounds,
        "series": list(labels),
        "summary": [
            f"{len(dimensions)} dimensions",
            f"{len(images)} graphs",
        ],
    }
