"""The simplest possible algorithm of Intrinsically Motivated Goal Exploration Processes
"""
from functools import partial
from typing import Any, Dict

from adtool.explorers.IMGEPExplorer import IMGEPExplorerInstance as BaseIMGEPExplorerInstance
from adtool.systems import System
from adtool.wrappers.mutators import add_gaussian_noise, call_mutate_method
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import ObjectSpec, instantiate_object, object_spec
from adtool.utils.leaf.Leaf import Leaf
from pydantic import Field
from pydantic import BaseModel

import numpy as np

from enum import Enum

class MutatorEnum(Enum):
    Gaussian = 'gaussian'
    Specific = 'specific'






class IMGEPConfig(BaseModel):
    equil_time: int = Field(1, ge=1, le=1000)
    behavior_map: ObjectSpec = Field(
        object_spec("adtool.maps.MeanBehaviorMap.MeanBehaviorMap")
    )
    parameter_map: ObjectSpec = Field(
        object_spec("adtool.maps.UniformParameterMap.UniformParameterMap")
    )
    mutator: MutatorEnum = Field(MutatorEnum.Specific)
    mutator_config: Dict = Field({})



class IMGEPExplorerInstance(BaseIMGEPExplorerInstance):
    def suggest_trial(
        self,
        lookback_length: int = -1,
        goal: np.ndarray = None,
        goal_targeting: Dict[str, Any] | None = None,
    ):
        if goal is None:
            goal = self.behavior_map.sample()

        interpolated_policy = self._vector_search_for_goal(goal, lookback_length)

        params_trial = self.mutator(interpolated_policy)

        return params_trial

    def _interpolate_policies_recursive(self, policy1, policy2, weight: float):
        
        if isinstance(policy1, np.ndarray):
            # element-wise interpolation
            return np.add((1 - weight) * policy1, weight * policy2)

        if isinstance(policy1, dict):
            interpolated_policy = {}
            for key in policy1.keys():
                interpolated_policy[key] = self._interpolate_policies_recursive(policy1[key], policy2[key], weight)
            return interpolated_policy
        elif isinstance(policy1, list):
            interpolated_policy = []
            for i in range(len(policy1)):
                interpolated_policy.append(self._interpolate_policies_recursive(policy1[i], policy2[i], weight))

            return interpolated_policy
        else:
            return (1 - weight) * policy1 + weight * policy2
    
    # same but also consider lists
    def _interpolate_policies(self, policy1: Dict, policy2: Dict, weight: float):

        dynamic_params= self._interpolate_policies_recursive(policy1['dynamic_params'], policy2['dynamic_params'], weight)
        return {'dynamic_params': dynamic_params}
                    
    def _vector_search_for_goal(self, goal: np.ndarray, lookback_length: int) -> Dict:
        goal_history, param_history = self._get_history_features(lookback_length)
        if goal_history.shape[0] == 0:
            return self.parameter_map.sample()

        closest_indices = self._nearest_indices(
            np.asarray(goal, dtype=float),
            lookback_length,
            k=2,
        )

        policy1 = param_history[closest_indices[0]]
        if len(closest_indices) == 1:
            return policy1
        policy2 = param_history[closest_indices[1]]

        # Calculate the weights based on the distances
        distances = np.linalg.norm(goal_history[closest_indices] - goal, axis=1)
        total_distance = np.sum(distances)
        weight = distances[0] / total_distance


        interpolated_policy = self._interpolate_policies(policy1, policy2, weight)

        return interpolated_policy

@expose
class IMGEPExplorer():
    config=IMGEPConfig

    # create specification for discovery attributes
    # TODO: kind of hard-coded for now, based on constructor defaults
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        pass


        
    def __call__(self,system) -> "IMGEPExplorerInstance":
        behavior_map = self.make_behavior_map(system)
        param_map = self.make_parameter_map(system)
        mutator = self.make_mutator(param_map)
        equil_time = self.config.equil_time
        explorer = IMGEPExplorerInstance(
            parameter_map=param_map,
            behavior_map=behavior_map,
            equil_time=equil_time,
            mutator=mutator,
        )

        return explorer

    def make_behavior_map(self, system: System):
        return instantiate_object(
            self.config.behavior_map,
            system,
            object_name="behavior map",
        )

    def make_parameter_map(self, system: System):
        return instantiate_object(
            self.config.parameter_map,
            system,
            object_name="parameter map",
        )

    def make_mutator(self, param_map: Any = None):
        if self.config.mutator== MutatorEnum.Specific:
            mutator = partial(call_mutate_method, param_map=param_map)
        elif self.config.mutator == MutatorEnum.Gaussian:
            mutator = partial(
                add_gaussian_noise, std=self.config.mutator_config["std"]
            )
        else:
            mutator =  None

        return mutator
    
