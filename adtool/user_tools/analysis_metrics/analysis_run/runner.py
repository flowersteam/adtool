from pathlib import Path

from .config import (
    load_analysis_run_config,
)
from ..shared import (
    AnalysisRunSummary,
    DatasetInfo,
    create_run_dir,
    load_analysis_module,
    load_discovery_set,
    write_summary,
)


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
    if not config.analysis_modules:
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
    for spec in config.analysis_modules:
        module = load_analysis_module(spec)
        module_key = module.identifier
        module_order.append(module_key)
        modules[module_key] = module.run(datasets, labels, run_dir)

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
