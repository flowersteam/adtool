"""Optional acceleration components for :class:`HistoryStore`."""

from .base import HistoryOptimization
from .exact_numpy import ExactNumpyHistoryOptimization

__all__ = ["ExactNumpyHistoryOptimization", "HistoryOptimization"]
