"""Compare coverage between two discovery folders."""

from .comparison import (
    ComparisonConfig,
    CoverageComparisonSummary,
    DimensionPretreatmentConfig,
    DiscoverySet,
    PlotConfig,
    compare_discovery_sets,
    load_comparison_config,
)

run_coverage_comparison = compare_discovery_sets

__all__ = [
    "CoverageComparisonSummary",
    "ComparisonConfig",
    "DimensionPretreatmentConfig",
    "DiscoverySet",
    "PlotConfig",
    "compare_discovery_sets",
    "load_comparison_config",
    "run_coverage_comparison",
]
