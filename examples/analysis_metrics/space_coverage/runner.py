from adtool.examples.analysis_metrics.shared import (
    AnalysisImage,
    apply_projection,
    order_sequence_by_run_idx,
)
from adtool.examples.analysis_metrics.space_coverage.metric import (
    load_space_coverage_metric,
)
from adtool.examples.analysis_metrics.space_coverage.plotting import (
    plot_progression_curves,
)


def _display_dimension_label(label, dim_index):
    return f"{label} ({dim_index})"


def _progression_bounds(series):
    x_values = [value for steps, _, _ in series for value in steps]
    y_values = [value for _, counts, _ in series for value in counts]
    return (
        (float(min(x_values)), float(max(x_values))),
        (float(min(y_values)), float(max(y_values))),
    )


def _progression_payload(title, y_label, dimensions, dimension_labels, metric_path, run_indices, counts, details, label):
    return {
        "label": label,
        "title": title,
        "y_label": y_label,
        "dimensions": list(dimensions),
        "dimension_labels": list(dimension_labels),
        "metric_path": metric_path,
        "steps": [int(run_idx) + 1 for run_idx in run_indices.tolist()],
        "counts": [int(value) for value in counts],
        "run_indices": [int(value) for value in run_indices.tolist()],
        "boundaries": [list(bounds) for bounds in details.get("boundaries", [])],
        "bins_per_dimension": list(details.get("bins_per_dimension", [])),
        "total_cells": details.get("total_cells"),
    }


def run_space_coverage(config, datasets, labels, run_dir):
    projected_values, raw_labels = apply_projection(config.projection, datasets)
    metric = load_space_coverage_metric(config.metric)

    progressions = []
    for dataset, values, label in zip(datasets, projected_values, labels):
        ordered_values, run_indices = order_sequence_by_run_idx(
            values,
            dataset.payloads,
            dataset.files,
        )
        counts, details = metric.compute_progression(ordered_values)
        progressions.append((ordered_values, run_indices, counts, details, label))

    first_values, _, _, first_details, _ = progressions[0]
    dimensions = list(first_details.get("dimensions", range(first_values.shape[1])))
    raw_labels = raw_labels or [f"dim_{idx}" for idx in range(first_values.shape[1])]
    dimension_labels = [
        _display_dimension_label(raw_labels[idx], idx)
        for idx in dimensions
    ]
    title = str(first_details.get("title", metric.title))
    y_label = str(first_details.get("y_label", metric.y_label))

    series = [
        ([int(run_idx) + 1 for run_idx in run_indices.tolist()], counts, label)
        for _, run_indices, counts, _, label in progressions
    ]
    x_bounds, y_bounds = _progression_bounds(series)
    image_name = f"space_coverage_progression.{config.plot.output_format}"
    plot_progression_curves(
        run_dir / image_name,
        series,
        title,
        y_label,
        config.plot,
    )

    images = [
        AnalysisImage(
            file=image_name,
            title=title,
            plot_type="space coverage progression",
            dimensions=dimensions,
            bounds=[x_bounds, y_bounds],
        ).to_payload()
    ]
    return {
        "title": title,
        "images": images,
        "progression": {
            "datasets": [
                _progression_payload(
                    title,
                    y_label,
                    dimensions,
                    dimension_labels,
                    config.metric.path,
                    run_indices,
                    counts,
                    details,
                    label,
                )
                for _, run_indices, counts, details, label in progressions
            ],
        },
        "series": list(labels),
        "summary": [
            f"{len(progressions)} datasets",
            f"{len(series[0][0]) if series else 0} steps",
            f"{len(images)} graph",
        ],
    }
