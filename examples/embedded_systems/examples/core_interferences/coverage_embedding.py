from typing import Any, Dict

import numpy as np


class CoreInterferenceCountEmbeddingBuilder:
    def __init__(self) -> None:
        pass

    def build(self, data: Dict[str, Any]) -> np.ndarray:
        raw_output = data.get("raw_output", {})
        mutual = raw_output.get("mutual", {}) if isinstance(raw_output, dict) else {}

        miss_core0 = self._sum_array(mutual.get("miss_core0"))
        miss_core1 = self._sum_array(mutual.get("miss_core1"))
        hits_core0 = self._sum_array(mutual.get("hits_core0"))
        hits_core1 = self._sum_array(mutual.get("hits_core1"))

        return np.array([miss_core0, miss_core1, hits_core0, hits_core1], dtype=float)

    @staticmethod
    def _sum_array(value: Any) -> float:
        if value is None:
            return 0.0
        return float(np.asarray(value, dtype=float).sum())
