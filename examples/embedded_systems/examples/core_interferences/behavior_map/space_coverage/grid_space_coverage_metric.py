from __future__ import annotations

from math import prod

import numpy as np

from adtool.examples.analysis_metrics.space_coverage import SpaceCoverageMetric


class GridSpaceCoverageMetric(SpaceCoverageMetric):
    def __init__(
        self,
        boundaries,
        bins_per_dimension,
        dimensions=None,
        title: str = "Space coverage progression",
        y_label: str = "covered cells",
    ) -> None:
        self.boundaries = np.asarray(boundaries, dtype=float)
        self.dimensions = None if dimensions is None else [int(dim) for dim in dimensions]
        self.title = title
        self.y_label = y_label
        dim_count = self.boundaries.shape[0]
        if isinstance(bins_per_dimension, int):
            self.bins_per_dimension = [int(bins_per_dimension)] * dim_count
        else:
            self.bins_per_dimension = [int(bins) for bins in bins_per_dimension]

    def compute_progression(
        self,
        points: np.ndarray,
    ):
        matrix = np.asarray(points, dtype=float)
        selected_dimensions = self._selected_dimensions(matrix.shape[1])
        selected_points = matrix[:, selected_dimensions]

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

    def _point_to_cell(self, point: np.ndarray):
        cell = []
        for value, bounds, bins in zip(point, self.boundaries, self.bins_per_dimension):
            low = float(bounds[0])
            high = float(bounds[1])
            if value < low or value > high:
                return None
            if value == high:
                cell.append(bins - 1)
                continue

            ratio = (value - low) / (high - low)
            cell.append(min(bins - 1, max(0, int(np.floor(ratio * bins)))))

        return tuple(cell)
