from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from examples.embedded_systems.behavior_encoders.base_behavior_encoder import (
    BaseBehaviorEncoder,
)
from examples.embedded_systems.goal_samplers.base_goal_sampler import BaseGoalSampler


class BaseBehaviorMap(Leaf):
    """Base behavior map with history tracking and goal sampling."""

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "output",
        goal_sampler: Optional[BaseGoalSampler] = None,
        behavior_encoder: Optional[BaseBehaviorEncoder] = None,
    ) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.goal_sampler = goal_sampler
        self.behavior_encoder = behavior_encoder
        self._history: List[np.ndarray] = []
        self._feature_size: Optional[int] = None
        self._history_min: Optional[np.ndarray] = None
        self._history_max: Optional[np.ndarray] = None

    def encode(self, raw_output: Dict[str, Any]) -> np.ndarray:
        if self.behavior_encoder is None:
            raise ValueError("Behavior encoder is not configured.")
        return self.behavior_encoder.encode(raw_output)

    def map(self, input: Dict) -> Dict:
        intermed = deepcopy(input)

        raw_output = intermed[self.premap_key]
        embedding = self.encode(raw_output)

        intermed["raw_" + self.premap_key] = raw_output
        del intermed[self.premap_key]
        intermed[self.postmap_key] = embedding

        self._history.append(embedding)
        if self._feature_size is None:
            self._feature_size = embedding.size
        if self._history_min is None or self._history_max is None:
            self._history_min = embedding.copy()
            self._history_max = embedding.copy()
        else:
            np.minimum(self._history_min, embedding, out=self._history_min)
            np.maximum(self._history_max, embedding, out=self._history_max)
        return intermed

    def sample(self) -> np.ndarray:
        if self.goal_sampler is None:
            if self._feature_size is None:
                return np.zeros(1, dtype=float)
            return np.zeros(self._feature_size, dtype=float)
        return self.goal_sampler.sample(
            self._history,
            self._feature_size,
            min_=self._history_min,
            max_=self._history_max,
        )
