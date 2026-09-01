import numpy as np

from ..shared import (
    AnalysisImage,
    apply_projection,
    branch_labels,
    displayed_branch_color,
    displayed_branch_id,
    order_sequence_by_run_idx,
)
from .metric import (
    load_space_coverage_metric,
)
from .plotting import (
    plot_progression_curves,
)


def _display_dimension_label(label, dim_index):
    return f"{label} ({dim_index})"


def _progression_bounds(series):
    x_values = [value for item in series for value in item["steps"]]
    y_values = [value for item in series for value in item["counts"]]
    return (
        (float(min(x_values)), float(max(x_values))),
        (float(min(y_values)), float(max(y_values))),
    )


def _progression_payload(
    title,
    y_label,
    dimensions,
    dimension_labels,
    metric_path,
    run_indices,
    counts,
    details,
    label,
    ordered_branch_ids,
    checkpoints,
):
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
        "branch_ids": list(ordered_branch_ids),
        "checkpoints": [
            {"step": int(step), "branch_id": branch_id}
            for step, branch_id in checkpoints
        ],
        "boundaries": [list(bounds) for bounds in details.get("boundaries", [])],
        "bins_per_dimension": list(details.get("bins_per_dimension", [])),
        "total_cells": details.get("total_cells"),
    }


def _ordered_branch_ids(dataset):
    run_indices = np.asarray([
        int((payload.get("metadata") or {})["run_idx"])
        for payload in dataset.payloads
    ])
    order = np.argsort(run_indices, kind="stable")
    return [
        str((dataset.payloads[index].get("metadata") or {}).get("branch_id") or "")
        for index in order
    ]


def _colored_progression(steps, counts, ordered_branch_ids, dataset, input_label, labels_by_branch):
    if not dataset.checkpoints:
        return {
            "steps": steps,
            "counts": counts,
            "segments": [(steps, counts, input_label, None, dataset.color)],
            "checkpoints": [],
        }

    displayed_branch_ids = [
        displayed_branch_id(dataset, branch_id)
        for branch_id in ordered_branch_ids
    ]
    segments = []
    start = 0
    for index in range(1, len(displayed_branch_ids) + 1):
        if index < len(displayed_branch_ids) and displayed_branch_ids[index] == displayed_branch_ids[start]:
            continue
        segment_start = max(0, start - 1)
        branch_id = displayed_branch_ids[start]
        segments.append(
            (
                steps[segment_start:index],
                counts[segment_start:index],
                labels_by_branch[branch_id],
                branch_id,
                displayed_branch_color(dataset, branch_id),
            )
        )
        start = index

    count_by_step = dict(zip(steps, counts))
    checkpoint_points = [
        (
            checkpoint.step,
            count_by_step[checkpoint.step],
            labels_by_branch[displayed_branch_id(dataset, checkpoint.branch_id)],
            displayed_branch_id(dataset, checkpoint.branch_id),
            displayed_branch_color(
                dataset,
                displayed_branch_id(dataset, checkpoint.branch_id),
            ),
        )
        for checkpoint in dataset.checkpoints
        if checkpoint.step in count_by_step
    ]
    plot_steps = list(steps)
    plot_counts = list(counts)
    if segments and plot_steps and plot_steps[0] > 0:
        first_steps, first_counts, first_label, first_branch_id, first_color = segments[0]
        segments[0] = (
            [0, *first_steps],
            [0, *first_counts],
            first_label,
            first_branch_id,
            first_color,
        )
        plot_steps.insert(0, 0)
        plot_counts.insert(0, 0)
    return {
        "steps": plot_steps,
        "counts": plot_counts,
        "segments": segments,
        "checkpoints": checkpoint_points,
    }


def run_space_coverage(config, datasets, labels, run_dir):
    projected_values, raw_labels = apply_projection(config.projection, datasets)
    metric = load_space_coverage_metric(config.metric)

    labels_by_branch = branch_labels(datasets, labels)
    progressions = []
    for dataset, values, label in zip(datasets, projected_values, labels):
        ordered_values, run_indices = order_sequence_by_run_idx(
            values,
            dataset.payloads,
            dataset.files,
        )
        counts, details = metric.compute_progression(ordered_values)
        ordered_branch_ids = _ordered_branch_ids(dataset)
        progressions.append(
            (
                ordered_values,
                run_indices,
                counts,
                details,
                label,
                ordered_branch_ids,
            )
        )

    first_values, _, _, first_details, _, _ = progressions[0]
    dimensions = list(first_details.get("dimensions", range(first_values.shape[1])))
    raw_labels = raw_labels or [f"dim_{idx}" for idx in range(first_values.shape[1])]
    dimension_labels = [
        _display_dimension_label(raw_labels[idx], idx)
        for idx in dimensions
    ]
    title = str(first_details.get("title", metric.title))
    y_label = str(first_details.get("y_label", metric.y_label))

    series = [
        _colored_progression(
            [int(run_idx) + 1 for run_idx in run_indices.tolist()],
            counts,
            ordered_branch_ids,
            dataset,
            label,
            labels_by_branch,
        )
        for dataset, (_, run_indices, counts, _, label, ordered_branch_ids) in zip(
            datasets,
            progressions,
        )
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
                    ordered_branch_ids,
                    [
                        (checkpoint.step, checkpoint.branch_id)
                        for checkpoint in dataset.checkpoints
                    ],
                )
                for dataset, (
                    _,
                    run_indices,
                    counts,
                    details,
                    label,
                    ordered_branch_ids,
                ) in zip(datasets, progressions)
            ],
        },
        "series": list(labels),
        "summary": [
            f"{len(progressions)} datasets",
            f"{len(progressions[0][1]) if progressions else 0} steps",
            f"{len(images)} graph",
        ],
    }
