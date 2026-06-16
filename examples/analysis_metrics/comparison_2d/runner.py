import numpy as np

from adtool.examples.analysis_metrics.comparison_2d.plotting import (
    plot_dimension_pair_scatter,
)
from adtool.examples.analysis_metrics.shared import AnalysisImage, apply_projection


def _display_dimension_label(label, dim_index):
    return f"{label} ({dim_index})"


def _value_bounds(values_a, values_b):
    min_value = min(float(np.min(values_a)), float(np.min(values_b)))
    max_value = max(float(np.max(values_a)), float(np.max(values_b)))
    if min_value == max_value:
        return min_value - 1.0, max_value + 1.0
    return min_value, max_value


def run_comparison_2d(config, dataset_a, dataset_b, label_a, label_b, run_dir):
    values_a, values_b, raw_labels = apply_projection(
        config.projection,
        dataset_a,
        dataset_b,
    )
    dim_count = values_a.shape[1]
    raw_labels = raw_labels or [f"dim_{idx}" for idx in range(dim_count)]

    images = []
    bounds = []
    pairs = []
    for x_dim, y_dim in config.pairs:
        x_values_a = values_a[:, x_dim]
        x_values_b = values_b[:, x_dim]
        y_values_a = values_a[:, y_dim]
        y_values_b = values_b[:, y_dim]
        x_label = _display_dimension_label(raw_labels[x_dim], x_dim)
        y_label = _display_dimension_label(raw_labels[y_dim], y_dim)
        image_name = f"comparison_2d_dims_{x_dim}_{y_dim}.{config.plot.output_format}"
        pair_bounds = [
            _value_bounds(x_values_a, x_values_b),
            _value_bounds(y_values_a, y_values_b),
        ]
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
        images.append(
            AnalysisImage(
                file=image_name,
                title=f"X = {x_label} | Y = {y_label}",
                plot_type="2d scatter",
                dimensions=[x_dim, y_dim],
                bounds=pair_bounds,
            ).to_payload()
        )
        bounds.append([list(pair_bounds[0]), list(pair_bounds[1])])
        pairs.append([x_dim, y_dim])

    return {
        "title": "2D comparison",
        "images": images,
        "pairs": pairs,
        "bounds": bounds,
    }
