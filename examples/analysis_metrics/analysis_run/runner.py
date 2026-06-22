from pathlib import Path

from adtool.examples.analysis_metrics.comparison_1d import run_comparison_1d
from adtool.examples.analysis_metrics.comparison_2d import run_comparison_2d
from adtool.examples.analysis_metrics.analysis_run.config import (
    load_analysis_run_config,
)
from adtool.examples.analysis_metrics.shared import (
    AnalysisRunSummary,
    DatasetInfo,
    create_run_dir,
    load_discovery_set,
    write_summary,
)
from adtool.examples.analysis_metrics.space_coverage import run_space_coverage


DEFAULT_OUTPUT_DIR = Path("analysis_runs")


def _default_label(path):
    return path.name or str(path)


def run_analysis(
    primary_path,
    comparison_paths,
    output_dir=DEFAULT_OUTPUT_DIR,
    primary_label=None,
    comparison_labels=None,
    config_file=None,
):
    config = load_analysis_run_config(config_file)
    if (
        config.comparison_1d is None
        and config.comparison_2d is None
        and config.space_coverage is None
    ):
        raise ValueError("No analysis module configured")

    primary_path = Path(primary_path).resolve()
    comparison_paths = [Path(path).resolve() for path in comparison_paths]
    if not comparison_paths:
        raise ValueError("At least one comparison dataset is required")

    labels = [primary_label or _default_label(primary_path)]
    comparison_labels = list(comparison_labels or [])
    while len(comparison_labels) < len(comparison_paths):
        comparison_labels.append(_default_label(comparison_paths[len(comparison_labels)]))
    labels.extend(
        label or _default_label(path)
        for label, path in zip(comparison_labels[:len(comparison_paths)], comparison_paths)
    )

    datasets = [load_discovery_set(primary_path)]
    datasets.extend(load_discovery_set(path) for path in comparison_paths)
    run_dir = create_run_dir(output_dir)

    module_order = []
    modules = {}
    if config.comparison_1d is not None:
        module_order.append("comparison_1d")
        modules["comparison_1d"] = run_comparison_1d(
            config.comparison_1d,
            datasets,
            labels,
            run_dir,
        )
    if config.comparison_2d is not None:
        module_order.append("comparison_2d")
        modules["comparison_2d"] = run_comparison_2d(
            config.comparison_2d,
            datasets,
            labels,
            run_dir,
        )
    if config.space_coverage is not None:
        module_order.append("space_coverage")
        modules["space_coverage"] = run_space_coverage(
            config.space_coverage,
            datasets,
            labels,
            run_dir,
        )

    summary = AnalysisRunSummary(
        run_dir=run_dir,
        datasets=[
            DatasetInfo(
                path=primary_path,
                label=labels[0],
                count=len(datasets[0].payloads),
                role="primary",
            ),
            *[
                DatasetInfo(
                    path=path,
                    label=label,
                    count=len(dataset.payloads),
                    role="comparison",
                )
                for path, label, dataset in zip(
                    comparison_paths,
                    labels[1:],
                    datasets[1:],
                )
            ],
        ],
        module_order=module_order,
        modules=modules,
    )
    write_summary(summary)
    return summary
