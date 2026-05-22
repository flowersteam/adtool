"""Coverage comparison utilities and pipeline."""

from .config import CoverageConfig, CoverageRunSummary, PlotConfig, load_coverage_config
from .embedding_builder import DefaultEmbeddingBuilder, build_embedding_builder


def run_coverage_comparison(config_path):
    from .pipeline import run_coverage_comparison as _run_coverage_comparison

    return _run_coverage_comparison(config_path)

__all__ = [
    "CoverageConfig",
    "CoverageRunSummary",
    "PlotConfig",
    "DefaultEmbeddingBuilder",
    "build_embedding_builder",
    "load_coverage_config",
    "run_coverage_comparison",
]
