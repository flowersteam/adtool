from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pydoc import locate
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SpaceCoverageMetricConfig:
    path: str
    config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SpaceCoverageProgressionSummary:
    title: str
    y_label: str
    steps: list
    counts: list
    run_indices: list
    dimensions: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)
    bins_per_dimension: list = field(default_factory=list)
    total_cells: Optional[int] = None

    def to_payload(self) -> dict:
        return {
            "title": self.title,
            "y_label": self.y_label,
            "steps": list(self.steps),
            "counts": list(self.counts),
            "run_indices": list(self.run_indices),
            "dimensions": list(self.dimensions),
            "boundaries": [list(bounds) for bounds in self.boundaries],
            "bins_per_dimension": list(self.bins_per_dimension),
            "total_cells": self.total_cells,
        }


class SpaceCoverageMetric(ABC):
    title = "Space coverage progression"
    y_label = "covered cells"

    @abstractmethod
    def compute_progression(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return cumulative space coverage counts and a metric description."""


def _locate_metric_class(path: str):
    metric_cls = locate(path)
    if metric_cls is None and path.startswith("adtool.examples."):
        metric_cls = locate(f"examples.{path[len('adtool.examples.'):]}")
    if metric_cls is None and path.startswith("examples."):
        metric_cls = locate(f"adtool.{path}")
    if metric_cls is None:
        raise ValueError(f"Could not retrieve space coverage metric from path: {path}")
    return metric_cls


def load_space_coverage_metric(
    config: SpaceCoverageMetricConfig,
) -> SpaceCoverageMetric:
    metric_cls = _locate_metric_class(config.path)
    return metric_cls(**dict(config.config or {}))


def _ordered_run_indices(
    payloads: list,
    files: list,
) -> np.ndarray:
    run_indices = []
    for file_path, payload in zip(files, payloads):
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or "run_idx" not in metadata:
            raise ValueError(
                "Space coverage progression requires discovery metadata.run_idx "
                f"in {file_path}"
            )
        run_indices.append(int(metadata["run_idx"]))
    return np.asarray(run_indices, dtype=int)


def _coerce_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("Space coverage progression expects a 2D value matrix")
    return matrix


def compute_space_coverage_progression(
    metric: SpaceCoverageMetric,
    values: np.ndarray,
    payloads: list,
    files: list,
) -> SpaceCoverageProgressionSummary:
    matrix = _coerce_matrix(values)
    if matrix.shape[0] != len(payloads) or len(payloads) != len(files):
        raise ValueError(
            "Space coverage progression requires one value row and one file per payload"
        )

    run_indices = _ordered_run_indices(payloads, files)
    order = np.argsort(run_indices, kind="stable")
    ordered_values = matrix[order]
    ordered_run_indices = run_indices[order]

    counts, details = metric.compute_progression(ordered_values)
    counts = np.asarray(counts, dtype=int).reshape(-1)
    if counts.size != ordered_values.shape[0]:
        raise ValueError(
            "Space coverage metric returned a progression with the wrong length"
        )

    dimensions = details.get("dimensions")
    if dimensions is None:
        dimensions = list(range(ordered_values.shape[1]))

    boundaries = details.get("boundaries") or []
    bins_per_dimension = details.get("bins_per_dimension") or []
    total_cells = details.get("total_cells")

    return SpaceCoverageProgressionSummary(
        title=str(details.get("title", metric.title)),
        y_label=str(details.get("y_label", metric.y_label)),
        steps=[int(run_idx) + 1 for run_idx in ordered_run_indices.tolist()],
        counts=counts.tolist(),
        run_indices=ordered_run_indices.tolist(),
        dimensions=[int(dim) for dim in dimensions],
        boundaries=[list(bounds) for bounds in boundaries],
        bins_per_dimension=[int(bins) for bins in bins_per_dimension],
        total_cells=None if total_cells is None else int(total_cells),
    )
