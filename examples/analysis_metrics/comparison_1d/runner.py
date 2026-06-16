import numpy as np

from adtool.examples.analysis_metrics.comparison_1d.plotting import (
    density_curve,
    plot_density_curves,
)
from adtool.examples.analysis_metrics.shared import AnalysisImage, apply_projection


def _resolve_dimensions(dimensions, dim_count):
    if dimensions == "all":
        return list(range(dim_count))
    return [int(dim) for dim in dimensions]


def _display_dimension_label(label, dim_index):
    return f"{label} ({dim_index})"


def _value_bounds(values_a, values_b):
    min_value = min(float(np.min(values_a)), float(np.min(values_b)))
    max_value = max(float(np.max(values_a)), float(np.max(values_b)))
    if min_value == max_value:
        return min_value - 1.0, max_value + 1.0
    return min_value, max_value


def _plot_bounds(values_a, values_b):
    integer_values = bool(
        np.allclose(values_a, np.round(values_a))
        and np.allclose(values_b, np.round(values_b))
    )
    min_value = min(float(np.min(values_a)), float(np.min(values_b)))
    max_value = max(float(np.max(values_a)), float(np.max(values_b)))
    if integer_values:
        return True, (min_value - 0.5, max_value + 0.5)
    if min_value == max_value:
        return False, (min_value - 1.0, max_value + 1.0)
    return False, (min_value, max_value)


def run_comparison_1d(config, dataset_a, dataset_b, label_a, label_b, run_dir):
    values_a, values_b, raw_labels = apply_projection(
        config.projection,
        dataset_a,
        dataset_b,
    )
    dim_count = values_a.shape[1]
    raw_labels = raw_labels or [f"dim_{idx}" for idx in range(dim_count)]
    dimensions = _resolve_dimensions(config.dimensions, dim_count)

    images = []
    labels = []
    bounds = []
    for dim_index in dimensions:
        dim_label = _display_dimension_label(raw_labels[dim_index], dim_index)
        dim_values_a = values_a[:, dim_index]
        dim_values_b = values_b[:, dim_index]
        integer_values, dim_bounds = _plot_bounds(dim_values_a, dim_values_b)
        image_name = f"comparison_1d_dim_{dim_index}.{config.plot.output_format}"
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
        images.append(
            AnalysisImage(
                file=image_name,
                title=dim_label,
                plot_type="1d density",
                dimensions=[dim_index],
                bounds=[dim_bounds],
            ).to_payload()
        )
        labels.append(dim_label)
        bounds.append(list(_value_bounds(dim_values_a, dim_values_b)))

    return {
        "title": "1D comparison",
        "images": images,
        "dimensions": list(dimensions),
        "labels": labels,
        "bounds": bounds,
    }
