"""Compare coverage between two discovery folders."""

from ..space_coverage import SpaceCoverageMetricConfig
from .comparison import (
    ComparisonConfig,
    CoverageImageSummary,
    CoverageComparisonSummary,
    DimensionPretreatmentConfig,
    DiscoverySet,
    PlotConfig,
    compare_discovery_sets,
    load_comparison_config,
)

run_coverage_comparison = compare_discovery_sets

__all__ = [
    "CoverageImageSummary",
    "CoverageComparisonSummary",
    "ComparisonConfig",
    "DimensionPretreatmentConfig",
    "DiscoverySet",
    "PlotConfig",
    "SpaceCoverageMetricConfig",
    "compare_discovery_sets",
    "load_comparison_config",
    "run_coverage_comparison",
]
