from typing import Any, Dict, List, Optional

import numpy as np


class ZScoreMetricEncoder:
    """Encode interference simulator output and z-score the selected metrics.

    This is a lightweight, per-observation normalization step. It keeps the
    same output contract as the existing encoder while centering and scaling
    the selected metrics to comparable magnitudes.
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

    def encode(self, raw_output: Dict[str, Any]) -> np.ndarray:
        """Convert mutual statistics to a single standardized feature vector."""
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

        mean, std = metrics.mean(), metrics.std() or 1.0
        if std == 0.0:
            return np.zeros_like(metrics, dtype=float)
        
        normalized = (metrics - mean) / std
        return np.clip(normalized, -3.0, 3.0).astype(np.float32)
