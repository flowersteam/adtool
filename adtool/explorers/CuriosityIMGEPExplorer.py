"""Curiosity-driven IMGEP using the shared chunked history store."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, Field
from adtool.explorers.IMGEPExplorer import IMGEPExplorerInstance as BaseIMGEPExplorer
from adtool.systems import System
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import ObjectSpec, instantiate_object, object_spec


class IMGEPConfig(BaseModel):
    equil_time: int = Field(1, ge=1, le=1000)
    behavior_map: ObjectSpec = Field(
        object_spec("adtool.maps.MeanBehaviorMap.MeanBehaviorMap")
    )
    parameter_map: ObjectSpec = Field(
        object_spec("adtool.maps.UniformParameterMap.UniformParameterMap")
    )
    mutator: ObjectSpec = Field(
        object_spec("adtool.mutators.SpecificMutator")
    )
    novelty_weight: float = Field(0.5, ge=0, le=1)


class CuriosityDrivenIMGEP(BaseIMGEPExplorer):
    def __init__(self, *args, novelty_weight: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.novelty_weight = novelty_weight

    def sample_curious_goal(self, lookback_length: int):
        """Draw a historical behavior with streaming reservoir sampling.

        The old KD-tree cache loaded every history item.  We deliberately do
        not replace it with another nearest-neighbor index: retrieval remains
        chunked and exact where a nearest match is needed.
        """
        match = self.history.random(lookback_length=lookback_length)
        return match.feature if match is not None else self.behavior_map.sample()

    def suggest_trial(
        self,
        lookback_length: int = -1,
        goal: np.ndarray | None = None,
        goal_targeting: Dict[str, Any] | None = None,
    ):
        if goal is None:
            if np.random.rand() < self.novelty_weight:
                goal = self.sample_curious_goal(lookback_length)
            else:
                goal = self.behavior_map.sample(goal_targeting=goal_targeting)
        return self.mutator(
            self._vector_search_for_goal(goal, lookback_length),
            parameter_map=self.parameter_map,
        )


@expose
class IMGEPExplorer:
    config = IMGEPConfig
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, system) -> CuriosityDrivenIMGEP:
        parameter_map = instantiate_object(
            self.config.parameter_map, system, object_name="parameter map"
        )
        behavior_map = instantiate_object(
            self.config.behavior_map, system, object_name="behavior map"
        )
        mutator = instantiate_object(self.config.mutator, object_name="mutator")
        return CuriosityDrivenIMGEP(
            parameter_map=parameter_map,
            behavior_map=behavior_map,
            mutator=mutator,
            equil_time=self.config.equil_time,
            novelty_weight=self.config.novelty_weight,
        )
