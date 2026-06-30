from typing import Any, Dict, List, Optional

import numpy as np

from examples.program_based_systems.types import BehaviorEncoder


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
        observation_vec = []

        for key1 in ["mutual", "core0", "core1"]:
            if key1 not in raw_output:
                continue

            for key2 in raw_output[key1].keys():
                value = np.array(raw_output[key1][key2]).reshape((-1,))

                if key2 in self.selection:
                    observation_vec.append(value)

        metrics = np.concatenate(observation_vec)

        return metrics