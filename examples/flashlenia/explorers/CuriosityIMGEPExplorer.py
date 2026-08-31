"""FlashLenia's curiosity-driven IMGEP variant."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, Field

from adtool.explorers.IMGEPExplorer import IMGEPExplorerInstance as BaseIMGEPExplorer
from adtool.maps.behavior import BehaviorMap
from adtool.maps.parameter import ParameterMap
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import ObjectSpec, instantiate_object, object_spec


class CuriosityIMGEPConfig(BaseModel):
    equil_time: int = Field(1, ge=1, le=1000)
    behavior_map: ObjectSpec
    parameter_map: ObjectSpec
    mutator: ObjectSpec = Field(object_spec("adtool.mutators.SpecificMutator"))
    novelty_weight: float = Field(0.5, ge=0, le=1)


class CuriosityDrivenIMGEP(BaseIMGEPExplorer):
    """IMGEP that favors mutation parents from sparse behavior regions."""

    curiosity_neighbors = 10
    random_goal_probability = 0.1

    def __init__(self, *args, novelty_weight: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.novelty_weight = novelty_weight
        self.uncertainty_map: np.ndarray | None = None
        self._curiosity_goals: np.ndarray | None = None

    def update_uncertainty_map(
        self, history_lookback_length: int = -1
    ) -> np.ndarray | None:
        """Score historical behaviors by mean distance to their nearest peers."""

        goals = self.history.features(
            history_lookback_length=history_lookback_length
        )
        if goals.size == 0:
            self.uncertainty_map = None
            self._curiosity_goals = None
            return None

        neighbor_count = min(len(goals), self.curiosity_neighbors)
        uncertainty = np.empty(len(goals), dtype=float)
        for index, goal in enumerate(goals):
            distances = np.linalg.norm(goals - goal, axis=1)
            nearest = np.partition(distances, neighbor_count - 1)[:neighbor_count]
            uncertainty[index] = float(np.mean(nearest))

        self.uncertainty_map = uncertainty
        self._curiosity_goals = goals
        return uncertainty

    def sample_curious_goal(
        self,
        history_lookback_length: int = -1,
        goal_targeting: Dict[str, Any] | None = None,
        *,
        refresh: bool = True,
    ) -> np.ndarray:
        """Sample a sparse historical behavior, with occasional fresh exploration."""

        uncertainty = (
            self.update_uncertainty_map(history_lookback_length)
            if refresh
            else self.uncertainty_map
        )
        if (
            uncertainty is None
            or self._curiosity_goals is None
            or np.random.rand() < self.random_goal_probability
        ):
            return self._sample_goal(goal_targeting)

        total_uncertainty = float(np.sum(uncertainty))
        probabilities = (
            uncertainty / total_uncertainty
            if np.isfinite(total_uncertainty) and total_uncertainty > 0
            else None
        )
        index = int(np.random.choice(len(self._curiosity_goals), p=probabilities))
        return self._curiosity_goals[index].copy()

    def suggest_trial(
        self,
        history_lookback_length: int = -1,
        goal: np.ndarray | None = None,
        goal_targeting: Dict[str, Any] | None = None,
    ):
        uncertainty = self.update_uncertainty_map(history_lookback_length)
        if goal is None:
            if uncertainty is None or np.random.rand() < self.novelty_weight:
                goal = self._sample_goal(goal_targeting)
            else:
                goal = self.sample_curious_goal(
                    history_lookback_length,
                    goal_targeting=goal_targeting,
                    refresh=False,
                )
        return self.mutator(
            self._vector_search_for_goal(goal, history_lookback_length),
            parameter_map=self.parameter_map,
        )

    def _sample_goal(self, goal_targeting: Dict[str, Any] | None) -> np.ndarray:
        if goal_targeting is None:
            return self.behavior_map.sample()
        return self.behavior_map.sample(goal_targeting=goal_targeting)


@expose
class CuriosityIMGEPExplorer:
    config = CuriosityIMGEPConfig
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        # Configuration is initialized by @expose.
        pass

    def __call__(self, system) -> CuriosityDrivenIMGEP:
        parameter_map = instantiate_object(
            self.config.parameter_map, system, object_name="parameter map"
        )
        behavior_map = instantiate_object(
            self.config.behavior_map, system, object_name="behavior map"
        )
        mutator = instantiate_object(self.config.mutator, object_name="mutator")
        if not isinstance(parameter_map, ParameterMap):
            raise TypeError("FlashLenia parameter_map must implement ParameterMap")
        if not isinstance(behavior_map, BehaviorMap):
            raise TypeError("FlashLenia behavior_map must implement BehaviorMap")
        return CuriosityDrivenIMGEP(
            parameter_map=parameter_map,
            behavior_map=behavior_map,
            mutator=mutator,
            equil_time=self.config.equil_time,
            novelty_weight=self.config.novelty_weight,
        )
