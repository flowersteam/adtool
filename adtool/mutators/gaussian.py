"""NumPy Gaussian mutation strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from adtool.mutators.base import BaseMutator


class GaussianMutator(BaseMutator):
    """Add independent Gaussian noise to a NumPy-compatible policy."""

    def __init__(self, mean: Any = 0.0, std: Any = 1.0) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    def __call__(self, parameters: Any, *, parameter_map: Any = None) -> np.ndarray:
        values = np.asarray(parameters)
        noise = np.random.normal(loc=self.mean, scale=self.std, size=values.shape)
        return values + noise
