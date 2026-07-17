import numpy as np

from .plotting import (
    plot_dimension_pair_scatter,
)
from ..shared import AnalysisImage, apply_projection


def _display_dimension_label(label, dim_index):
    return f"{label} ({dim_index})"


def _value_bounds(series_values):
    min_value = min(float(np.min(values)) for values in series_values)
    max_value = max(float(np.max(values)) for values in series_values)
    if min_value == max_value:
        return min_value - 1.0, max_value + 1.0
    return min_value, max_value


def run_comparison_2d(config, datasets, labels, run_dir):
    projected_values, raw_labels = apply_projection(config.projection, datasets)
    dim_count = projected_values[0].shape[1]
    raw_labels = raw_labels or [f"dim_{idx}" for idx in range(dim_count)]

    images = []
    bounds = []
    pairs = []
    for x_dim, y_dim in config.pairs:
        x_series = [values[:, x_dim] for values in projected_values]
        y_series = [values[:, y_dim] for values in projected_values]
        x_label = _display_dimension_label(raw_labels[x_dim], x_dim)
        y_label = _display_dimension_label(raw_labels[y_dim], y_dim)
        image_name = f"comparison_2d_dims_{x_dim}_{y_dim}.{config.plot.output_format}"
        pair_bounds = [
            _value_bounds(x_series),
            _value_bounds(y_series),
        ]
        plot_dimension_pair_scatter(
            run_dir / image_name,
            [
                (x_values, y_values, label)
                for x_values, y_values, label in zip(x_series, y_series, labels)
            ],
            x_label,
            y_label,
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
        "series": list(labels),
        "summary": [
            f"{len(pairs)} pairs",
            f"{len(images)} graphs",
        ],
    }
