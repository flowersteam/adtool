"""Bounded numeric-space helper used by behavior and parameter maps."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from adtool.maps.base import BaseMap, Payload


class BoxProjector(BaseMap):
    """Clamp a payload value to a box and sample from observed bounds."""

    def __init__(
        self,
        premap_key: str,
        bound_upper: np.ndarray = np.array([float("inf")]),
        bound_lower: np.ndarray = np.array([float("-inf")]),
        init_low: Optional[np.ndarray] = None,
        init_high: Optional[np.ndarray] = None,
        tensor_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__(premap_key=premap_key, postmap_key=premap_key)
        self.bound_upper = np.asarray(bound_upper)
        self.bound_lower = np.asarray(bound_lower)
        self.tensor_shape = np.asarray(init_low).shape if init_low is not None else tensor_shape
        self.low = None if init_low is None else np.asarray(init_low, dtype=float)
        self.high = None if init_high is None else np.asarray(init_high, dtype=float)

    def map(self, payload: Payload, override_existing: bool = True) -> Payload:
        del override_existing
        output = deepcopy(payload)
        data = np.asarray(output[self.premap_key])
        if self.tensor_shape is None:
            self.tensor_shape = data.shape
        data = np.minimum(np.maximum(data, self.bound_lower), self.bound_upper)
        self._update_low_high(data)
        output[self.premap_key] = data
        return output

    def sample(self) -> np.ndarray:
        if self.tensor_shape is None or self.low is None or self.high is None:
            raise RuntimeError("BoxProjector must observe data before sampling.")
        weights = np.random.rand(*self.tensor_shape)
        with np.errstate(over="ignore", invalid="ignore"):
            lengths = self.high - self.low
        if np.isfinite(lengths).all():
            return weights * lengths + self.low
        # Preserve the historic sampling calculation when it is safe, but
        # avoid a range-overflow exception for extreme finite bounds.
        return (1.0 - weights) * self.low + weights * self.high

    def _update_low_high(self, data: np.ndarray) -> None:
        values = np.asarray(data)
        if self.low is None:
            self.low = np.zeros_like(values)
        if self.high is None:
            self.high = np.zeros_like(values)
        self.low = self.low.astype(np.float32)
        self.high = self.high.astype(np.float32)
        low_mask = np.less(values, self.low)
        high_mask = np.greater(values, self.high)
        self.low[low_mask] = values[low_mask]
        self.high[high_mask] = values[high_mask]
