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


def _selected_checkpoint_path(dataset):
    if not dataset.checkpoints:
        return None
    return dataset.checkpoints[-1].path.resolve()


def _deduplicate_dataset_entries(entries):
    """Keep each selected checkpoint once, preserving the input order."""
    resolved_entries = []
    seen_checkpoint_paths = set()
    for entry in entries:
        checkpoint_path = _selected_checkpoint_path(entry[2])
        if checkpoint_path is not None:
            if checkpoint_path in seen_checkpoint_paths:
                continue
            seen_checkpoint_paths.add(checkpoint_path)
        resolved_entries.append(entry)
    return resolved_entries


def run_analysis(
    discovery_paths,
    output_dir=DEFAULT_OUTPUT_DIR,
    labels=None,
    config_file=None,
):
    config = load_analysis_run_config(config_file)
    if not config.analysis_modules:
        raise ValueError("No analysis module configured")

    discovery_paths = [Path(path).resolve() for path in discovery_paths]
    if not discovery_paths:
        raise ValueError("At least one discoveries or checkpoint path is required")

    labels = list(labels or [])
    while len(labels) < len(discovery_paths):
        labels.append(_default_label(discovery_paths[len(labels)]))
    labels = [
        label or _default_label(path)
        for label, path in zip(labels[:len(discovery_paths)], discovery_paths)
    ]

    datasets = [
        load_discovery_set(path, checkpoint_name=config.checkpoint_name)
        for path in discovery_paths
    ]
    entries = _deduplicate_dataset_entries(
        list(zip(discovery_paths, labels, datasets))
    )
    datasets = [entry[2] for entry in entries]
    labels = [entry[1] for entry in entries]
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
                path=path,
                label=label,
                count=len(dataset.payloads),
            )
            for path, label, dataset in entries
        ],
        module_order=module_order,
        modules=modules,
    )
    write_summary(summary)
    return summary
