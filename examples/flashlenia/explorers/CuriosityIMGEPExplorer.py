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
    def __init__(self, *args, novelty_weight: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.novelty_weight = novelty_weight

    def sample_curious_goal(self, history_lookback_length: int):
        match = self.history.random(history_lookback_length=history_lookback_length)
        return match.feature if match is not None else self.behavior_map.sample()

    def suggest_trial(
        self,
        history_lookback_length: int = -1,
        goal: np.ndarray | None = None,
        goal_targeting: Dict[str, Any] | None = None,
    ):
        if goal is None:
            goal = (
                self.sample_curious_goal(history_lookback_length)
                if np.random.rand() < self.novelty_weight
                else self.behavior_map.sample(goal_targeting=goal_targeting)
            )
        return self.mutator(
            self._vector_search_for_goal(goal, history_lookback_length),
            parameter_map=self.parameter_map,
        )


@expose
class CuriosityIMGEPExplorer:
    config = CuriosityIMGEPConfig
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

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
