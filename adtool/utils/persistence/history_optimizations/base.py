"""Extension contract for optional history query accelerators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from adtool.utils.persistence.history import HistoryStore


class HistoryOptimization(ABC):
    """Optional persistence and unlimited-lookback query extension.

    Implementations are owned by an explorer configuration.  The default
    :class:`HistoryStore` neither creates nor depends on an optimization.
    """

    @abstractmethod
    def attach(self, history: "HistoryStore") -> None:
        """Bind the extension to its owning history store."""

    @abstractmethod
    def record(self, discovery: dict[str, Any]) -> None:
        """Observe a newly recorded discovery."""

    @abstractmethod
    def set_head(self, checkpoint_dir: Path) -> None:
        """Select and validate the persisted checkpoint branch."""

    @abstractmethod
    def write_checkpoint(
        self,
        directory: str | Path,
        *,
        checkpoint_name: str,
        history_files: list[str],
    ) -> dict[str, Any]:
        """Write extension files into a checkpoint temporary directory."""

    @abstractmethod
    def feature_bounds(
        self, history_lookback_length: int = -1
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return bounds for the requested chronological history tail."""

    @abstractmethod
    def nearest(
        self,
        goal: np.ndarray,
        *,
        k: int,
        history_lookback_length: int,
        normalized: bool,
        normalization_bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> list[Any]:
        """Return nearest matches from the requested chronological history tail."""
