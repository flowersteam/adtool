from typing import Any, Dict, List, Optional

import numpy as np

from adtool.examples.embedded_systems.types import BehaviorEncoder


class ZScoreMetricEncoder(BehaviorEncoder):
    """Encode interference simulator output with running global z-score.

    Mean and variance are updated online across all observations seen so far,
    so normalization uses training-history statistics rather than per-sample
    statistics.
    """

    def __init__(self, selection: Optional[List[str]] = None) -> None:
        if selection is None:
            selection = [
                "miss_core0",
                "miss_core1",
                "hits_core0",
                "hits_core1",
                "diff_time_core0",
                "diff_time_core1",
                "diff_time",
            ]
            selection += [
                f"L2_{c}_{type_}_{core}"
                for c in ["miss", "hit"]
                for type_ in ["write", "read"]
                for core in ["core0", "core1"]
            ]
        self.selection = selection
        self._count = 0
        self._mean: Optional[np.ndarray] = None
        self._m2: Optional[np.ndarray] = None

    def encode(self, raw_output: Dict[str, Any]) -> np.ndarray:
        """Convert mutual statistics to a globally standardized feature vector."""
        mutual = raw_output.get("mutual", {})
        observation_vec = []

        for key in self.selection:
            value = np.array(mutual.get(key, 0.0), dtype=float).reshape((-1,))
            observation_vec.append(value)

        if not observation_vec:
            return np.array([], dtype=float)

        metrics = np.concatenate(observation_vec)
        metrics = np.nan_to_num(
            metrics, nan=0.0, posinf=1.0, neginf=-1.0)

        if self._mean is None or self._m2 is None:
            self._mean = np.zeros_like(metrics, dtype=float)
            self._m2 = np.zeros_like(metrics, dtype=float)
        elif metrics.shape != self._mean.shape:
            raise ValueError(
                "ZScoreMetricEncoder received a metric vector with a different shape "
                f"({metrics.shape}) than previous observations ({self._mean.shape})."
            )

        # Vectorized Welford update for numerically stable online variance.
        self._count += 1
        delta = metrics - self._mean
        self._mean += delta / self._count
        delta2 = metrics - self._mean
        self._m2 += delta * delta2

        if self._count < 2:
            return np.zeros_like(metrics, dtype=np.float32)

        variance = self._m2 / self._count
        std = np.sqrt(np.maximum(variance, 1e-12))
        normalized = (metrics - self._mean) / std
        return np.clip(normalized, -3.0, 3.0).astype(np.float32)
