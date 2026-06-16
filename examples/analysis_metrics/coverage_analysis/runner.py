from pathlib import Path

from adtool.examples.analysis_metrics.comparison_1d import run_comparison_1d
from adtool.examples.analysis_metrics.comparison_2d import run_comparison_2d
from adtool.examples.analysis_metrics.coverage_analysis.config import (
    load_coverage_analysis_config,
)
from adtool.examples.analysis_metrics.shared import (
    CoverageAnalysisSummary,
    DatasetInfo,
    create_run_dir,
    load_discovery_set,
    write_summary,
)
from adtool.examples.analysis_metrics.space_coverage import run_space_coverage


DEFAULT_OUTPUT_DIR = Path("coverage_runs")


def _default_label(path):
    return path.name or str(path)


def run_coverage_analysis(
    discovery_a_path,
    discovery_b_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    label_a=None,
    label_b=None,
    config_file=None,
):
    config = load_coverage_analysis_config(config_file)
    if (
        config.comparison_1d is None
        and config.comparison_2d is None
        and config.space_coverage is None
    ):
        raise ValueError("No analysis module configured")

    discovery_a_path = Path(discovery_a_path).resolve()
    discovery_b_path = Path(discovery_b_path).resolve()
    label_a = label_a or _default_label(discovery_a_path)
    label_b = label_b or _default_label(discovery_b_path)

    dataset_a = load_discovery_set(discovery_a_path)
    dataset_b = load_discovery_set(discovery_b_path)
    run_dir = create_run_dir(output_dir)

    module_order = []
    modules = {}
    if config.comparison_1d is not None:
        module_order.append("comparison_1d")
        modules["comparison_1d"] = run_comparison_1d(
            config.comparison_1d,
            dataset_a,
            dataset_b,
            label_a,
            label_b,
            run_dir,
        )
    if config.comparison_2d is not None:
        module_order.append("comparison_2d")
        modules["comparison_2d"] = run_comparison_2d(
            config.comparison_2d,
            dataset_a,
            dataset_b,
            label_a,
            label_b,
            run_dir,
        )
    if config.space_coverage is not None:
        module_order.append("space_coverage")
        modules["space_coverage"] = run_space_coverage(
            config.space_coverage,
            dataset_a,
            dataset_b,
            label_a,
            label_b,
            run_dir,
        )

    summary = CoverageAnalysisSummary(
        run_dir=run_dir,
        dataset_a=DatasetInfo(
            path=discovery_a_path,
            label=label_a,
            count=len(dataset_a.payloads),
        ),
        dataset_b=DatasetInfo(
            path=discovery_b_path,
            label=label_b,
            count=len(dataset_b.payloads),
        ),
        module_order=module_order,
        modules=modules,
    )
    write_summary(summary)
    return summary
