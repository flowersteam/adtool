from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from examples.embedded_systems.systems.InterferenceSystem import InterferenceSystem
from examples.embedded_systems.helpers.module_factory import make_module


class InterferenceBehaviorMap(Leaf):
    """Behavior map."""

    def __init__(
            self,
            system: InterferenceSystem,
            premap_key: str = "output",
            postmap_key: str = "output",
            goal_sampler_config: Optional[Dict[str, Any]] = {
                "path": "examples.embedded_systems.goal_samplers.RandomMinMaxGoalSampler"
            },
            behavior_encoder_config: Optional[Dict[str, Any]] = {
                "path": "examples.embedded_systems.behavior_encoders.InterferenceMetricEncoder"
            },
    ) -> None:
        super().__init__()
        _ = system
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self._history: List[np.ndarray] = []
        self._feature_size = None
        self._history_min: Optional[np.ndarray] = None
        self._history_max: Optional[np.ndarray] = None
        self.goal_sampler = make_module(
            "goal_sampler",
            **goal_sampler_config,
        )
        self.behavior_encoder = make_module(
            "behavior_encoder",
            **behavior_encoder_config,
        )

    def map(self, input: Dict) -> Dict:
        intermed = deepcopy(input)

        raw_output = intermed[self.premap_key]
        embedding = self.behavior_encoder.encode(raw_output)

        intermed["raw_" + self.premap_key] = raw_output
        del intermed[self.premap_key]
        # `postmap_key` is what explorers treat as the behavior embedding used
        # for kNN retrieval and goal matching.
        intermed[self.postmap_key] = embedding

        self._history.append(embedding)
        # `_feature_size` allows safe cold-start goal sampling even before enough
        # history has accumulated.
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
        """Sample goals from behavior history."""
        return self.goal_sampler.sample(
            self._history,
            self._feature_size,
            min_=self._history_min,
            max_=self._history_max,
        )
