import json
from dataclasses import dataclass
from pathlib import Path

from adtool.examples.analysis_metrics.comparison_1d import load_comparison_1d_config
from adtool.examples.analysis_metrics.comparison_2d import load_comparison_2d_config
from adtool.examples.analysis_metrics.space_coverage import load_space_coverage_config


@dataclass(frozen=True)
class CoverageAnalysisConfig:
    comparison_1d: object = None
    comparison_2d: object = None
    space_coverage: object = None


def load_coverage_analysis_config(config_path):
    if config_path is None:
        return CoverageAnalysisConfig()

    with Path(config_path).open("r") as handle:
        payload = json.load(handle)

    return CoverageAnalysisConfig(
        comparison_1d=None
        if payload.get("comparison_1d") is None
        else load_comparison_1d_config(payload["comparison_1d"]),
        comparison_2d=None
        if payload.get("comparison_2d") is None
        else load_comparison_2d_config(payload["comparison_2d"]),
        space_coverage=None
        if payload.get("space_coverage") is None
        else load_space_coverage_config(payload["space_coverage"]),
    )
