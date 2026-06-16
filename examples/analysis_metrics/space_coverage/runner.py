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


def _progression_bounds(steps_a, counts_a, steps_b, counts_b):
    x_values = list(steps_a) + list(steps_b)
    y_values = list(counts_a) + list(counts_b)
    return (
        (float(min(x_values)), float(max(x_values))),
        (float(min(y_values)), float(max(y_values))),
    )


def _progression_payload(title, y_label, dimensions, dimension_labels, metric_path, run_indices, counts, details):
    return {
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


def run_space_coverage(config, dataset_a, dataset_b, label_a, label_b, run_dir):
    values_a, values_b, raw_labels = apply_projection(
        config.projection,
        dataset_a,
        dataset_b,
    )
    metric = load_space_coverage_metric(config.metric)

    ordered_values_a, run_indices_a = order_sequence_by_run_idx(
        values_a,
        dataset_a.payloads,
        dataset_a.files,
    )
    ordered_values_b, run_indices_b = order_sequence_by_run_idx(
        values_b,
        dataset_b.payloads,
        dataset_b.files,
    )

    counts_a, details_a = metric.compute_progression(ordered_values_a)
    counts_b, details_b = metric.compute_progression(ordered_values_b)

    dimensions = list(details_a.get("dimensions", range(ordered_values_a.shape[1])))
    raw_labels = raw_labels or [f"dim_{idx}" for idx in range(ordered_values_a.shape[1])]
    dimension_labels = [
        _display_dimension_label(raw_labels[idx], idx)
        for idx in dimensions
    ]
    title = str(details_a.get("title", metric.title))
    y_label = str(details_a.get("y_label", metric.y_label))

    steps_a = [int(run_idx) + 1 for run_idx in run_indices_a.tolist()]
    steps_b = [int(run_idx) + 1 for run_idx in run_indices_b.tolist()]
    x_bounds, y_bounds = _progression_bounds(steps_a, counts_a, steps_b, counts_b)
    image_name = f"space_coverage_progression.{config.plot.output_format}"
    plot_progression_curves(
        run_dir / image_name,
        steps_a,
        counts_a,
        steps_b,
        counts_b,
        title,
        label_a,
        label_b,
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
            "dataset_a": _progression_payload(
                title,
                y_label,
                dimensions,
                dimension_labels,
                config.metric.path,
                run_indices_a,
                counts_a,
                details_a,
            ),
            "dataset_b": _progression_payload(
                title,
                y_label,
                dimensions,
                dimension_labels,
                config.metric.path,
                run_indices_b,
                counts_b,
                details_b,
            ),
        },
    }
