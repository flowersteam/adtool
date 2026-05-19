"""Coverage comparison utilities and pipeline."""

from .pipeline import run_coverage_comparison
from .config import CoverageConfig, CoverageRunSummary, PlotConfig, load_coverage_config
from .embedding_builder import DefaultEmbeddingBuilder, build_embedding_builder

__all__ = [
    "CoverageConfig",
    "CoverageRunSummary",
    "PlotConfig",
    "DefaultEmbeddingBuilder",
    "build_embedding_builder",
    "load_coverage_config",
    "run_coverage_comparison",
]
