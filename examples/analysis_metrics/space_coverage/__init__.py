"""Shared abstractions and helpers for space-coverage metrics."""

from .metric import (
    SpaceCoverageMetric,
    SpaceCoverageMetricConfig,
    SpaceCoverageProgressionSummary,
    compute_space_coverage_progression,
    load_space_coverage_metric,
)

__all__ = [
    "SpaceCoverageMetric",
    "SpaceCoverageMetricConfig",
    "SpaceCoverageProgressionSummary",
    "compute_space_coverage_progression",
    "load_space_coverage_metric",
]
