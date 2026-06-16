from __future__ import annotations

from math import prod
from typing import Any, Dict, Optional

import numpy as np

from examples.analysis_metrics.space_coverage import SpaceCoverageMetric


class GridSpaceCoverageMetric(SpaceCoverageMetric):
    def __init__(
        self,
        boundaries,
        bins_per_dimension,
        dimensions: Optional[list] = None,
        title: str = "Space coverage progression",
        y_label: str = "covered cells",
    ) -> None:
        self.boundaries = np.asarray(boundaries, dtype=float)
        self.dimensions = None if dimensions is None else [int(dim) for dim in dimensions]
        self.title = title
        self.y_label = y_label

        if self.boundaries.ndim != 2 or self.boundaries.shape[1] != 2:
            raise ValueError("GridSpaceCoverageMetric boundaries must be shaped (n, 2)")

        dim_count = self.boundaries.shape[0]
        if isinstance(bins_per_dimension, int):
            self.bins_per_dimension = [int(bins_per_dimension)] * dim_count
        else:
            self.bins_per_dimension = [int(bins) for bins in bins_per_dimension]

        if len(self.bins_per_dimension) != dim_count:
            raise ValueError(
                "GridSpaceCoverageMetric bins_per_dimension must match boundaries"
            )

    def compute_progression(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        matrix = np.asarray(points, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("GridSpaceCoverageMetric expects a 2D point matrix")

        selected_dimensions = self._selected_dimensions(matrix.shape[1])
        selected_points = matrix[:, selected_dimensions]
        if selected_points.shape[1] != self.boundaries.shape[0]:
            raise ValueError(
                "GridSpaceCoverageMetric boundaries must match the selected space "
                f"({selected_points.shape[1]} dims)"
            )

        visited = set()
        counts = []
        for point in selected_points:
            cell = self._point_to_cell(point)
            if cell is not None:
                visited.add(cell)
            counts.append(len(visited))

        return np.asarray(counts, dtype=int), {
            "title": self.title,
            "y_label": self.y_label,
            "dimensions": selected_dimensions,
            "boundaries": self.boundaries.tolist(),
            "bins_per_dimension": list(self.bins_per_dimension),
            "total_cells": prod(self.bins_per_dimension),
        }

    def _selected_dimensions(self, dim_count: int) -> list:
        if self.dimensions is None:
            return list(range(dim_count))
        return list(self.dimensions)

    def _point_to_cell(self, point: np.ndarray) -> Optional[tuple]:
        cell = []
        for value, bounds, bins in zip(point, self.boundaries, self.bins_per_dimension):
            if not np.isfinite(value):
                return None

            low = float(bounds[0])
            high = float(bounds[1])
            if high <= low:
                raise ValueError(
                    "GridSpaceCoverageMetric requires strictly increasing boundaries"
                )
            if value < low or value > high:
                return None
            if bins < 1:
                raise ValueError(
                    "GridSpaceCoverageMetric requires positive bins_per_dimension"
                )

            if value == high:
                cell.append(bins - 1)
                continue

            ratio = (value - low) / (high - low)
            cell.append(min(bins - 1, max(0, int(np.floor(ratio * bins)))))

        return tuple(cell)
