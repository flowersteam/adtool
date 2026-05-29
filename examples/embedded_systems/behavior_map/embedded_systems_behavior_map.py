from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from adtool.maps.Map import Map
from adtool.examples.embedded_systems.behavior_map.encoder.embedded_systems_behavior_encoder import (
    BaseBehaviorEncoder,
)
from adtool.examples.embedded_systems.behavior_map.goal_sampler.embedded_systems_goal_sampler import (
    BaseGoalSampler,
)


class BaseBehaviorMap(Map):
    """Base behavior map with goal sampling."""

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "output",
        goal_sampler: Optional[BaseGoalSampler] = None,
        behavior_encoder: Optional[BaseBehaviorEncoder] = None,
    ) -> None:
        super().__init__(premap_key=premap_key, postmap_key=postmap_key)
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.goal_sampler = goal_sampler
        self.behavior_encoder = behavior_encoder

    def encode(self, raw_output: Dict[str, Any]) -> np.ndarray:
        if self.behavior_encoder is None:
            raise ValueError("Behavior encoder is not configured.")
        return self.behavior_encoder.encode(raw_output)

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        _ = override_existing
        intermed = deepcopy(input)

        raw_output = intermed[self.premap_key]
        embedding = self.encode(raw_output)

        intermed["raw_" + self.premap_key] = raw_output
        del intermed[self.premap_key]
        intermed[self.postmap_key] = embedding

        return intermed

    def sample(self, history: Optional[np.ndarray] = None) -> np.ndarray:
        if history is None:
            history_items = []
            feature_size = None
            min_ = None
            max_ = None
        else:
            history = np.asarray(history, dtype=float)
            if history.size == 0:
                history_items = []
                feature_size = None
                min_ = None
                max_ = None
            else:
                if history.ndim == 1:
                    history = history.reshape(1, -1)
                history_items = [row for row in history]
                feature_size = history.shape[1]
                min_ = history.min(axis=0)
                max_ = history.max(axis=0)

        if self.goal_sampler is None:
            if feature_size is None:
                return np.zeros(1, dtype=float)
            return np.zeros(feature_size, dtype=float)

        return self.goal_sampler.sample(
            history_items,
            feature_size,
            min_=min_,
            max_=max_,
        )
