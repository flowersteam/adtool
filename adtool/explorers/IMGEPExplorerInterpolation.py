"""IMGEP variant that interpolates the two nearest stored policies."""

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


class IMGEPExplorerInstance(BaseIMGEPExplorer):
    """The standard explorer with two-neighbor policy interpolation."""

    def _interpolate_policies_recursive(self, first: Any, second: Any, weight: float):
        if isinstance(first, np.ndarray):
            return (1 - weight) * first + weight * second
        if isinstance(first, dict):
            return {
                key: self._interpolate_policies_recursive(first[key], second[key], weight)
                for key in first
            }
        if isinstance(first, list):
            return [
                self._interpolate_policies_recursive(left, right, weight)
                for left, right in zip(first, second)
            ]
        return (1 - weight) * first + weight * second

    def _interpolate_policies(self, first: Dict, second: Dict, weight: float) -> Dict:
        return {
            "dynamic_params": self._interpolate_policies_recursive(
                first["dynamic_params"], second["dynamic_params"], weight
            )
        }

    def _vector_search_for_goal(self, goal: np.ndarray, lookback_length: int) -> Dict:
        matches = self.history.nearest(
            np.asarray(goal, dtype=float), k=2, lookback_length=lookback_length
        )
        if not matches:
            return self.parameter_map.sample()
        if len(matches) == 1:
            return matches[0].payload
        first, second = matches
        total_distance = first.distance**0.5 + second.distance**0.5
        if total_distance == 0:
            return first.payload
        return self._interpolate_policies(
            first.payload,
            second.payload,
            first.distance**0.5 / total_distance,
        )


@expose
class IMGEPExplorer:
    config = IMGEPConfig
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, system: System) -> IMGEPExplorerInstance:
        parameter_map = instantiate_object(
            self.config.parameter_map, system, object_name="parameter map"
        )
        behavior_map = instantiate_object(
            self.config.behavior_map, system, object_name="behavior map"
        )
        mutator = instantiate_object(self.config.mutator, object_name="mutator")
        return IMGEPExplorerInstance(
            parameter_map=parameter_map,
            behavior_map=behavior_map,
            mutator=mutator,
            equil_time=self.config.equil_time,
        )
