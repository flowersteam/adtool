from typing import Any, Dict, List, Optional

import numpy as np

from adtool.examples.embedded_systems.types import BehaviorEncoder


class InterferenceMetricEncoder(BehaviorEncoder):
    """Encode interference simulator output into a fixed feature vector."""

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
        mutual = raw_output.get("mutual", {})
        observation_vec = []

        for key in self.selection:
            value = np.array(mutual.get(key, 0.0), dtype=float).reshape((-1,))
            observation_vec.append(value)

        if not observation_vec:
            return np.array([], dtype=float)

        metrics = np.concatenate(observation_vec)
        metrics = np.nan_to_num(metrics, nan=0.0, posinf=0.0, neginf=0.0)
        return metrics.astype(float)
