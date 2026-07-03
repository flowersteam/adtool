import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from adtool.explorers.IMGEPExplorer import IMGEPExplorerInstance
from adtool.systems import System
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import ObjectSpec, instantiate_object, object_spec
from examples.program_based_systems.behavior_map.program_based_systems_behavior_map import (
    BaseBehaviorMap,
)
from examples.program_based_systems.parameter_map.program_based_systems_parameter_map import (
    BaseParameterMap,
)


class BaseExplorerFactory(ABC):
    """Factory interface that builds a runnable explorer instance."""

    discovery_spec: List[str] = []

    @abstractmethod
    def __call__(self, system: Any) -> Leaf:
        ...

class BaseExplorerConfig(BaseModel):
    periode: int = Field(1, ge=1, le=100000)
    knn: int = Field(1, ge=1, le=1000)
    behavior_map: ObjectSpec = Field(
        object_spec(
            "examples.program_based_systems.examples.core_interferences.behavior_map.InterferenceBehaviorMap"
        )
    )
    parameter_map: ObjectSpec = Field(
        object_spec(
            "examples.program_based_systems.examples.core_interferences.parameter_map.InterferenceParameterMap"
        )
    )

class BaseIMGEPInstance(IMGEPExplorerInstance):
    """Program-based systems IMGEP policy with periodic goals and kNN retrieval."""

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "params",
        parameter_map: BaseParameterMap | None = None,
        behavior_map: BaseBehaviorMap | None = None,
        periode: int = 1,
        knn: int = 1,
    ) -> None:
        if parameter_map is None or behavior_map is None:
            raise ValueError("BaseIMGEPInstance requires parameter_map and behavior_map.")
        super().__init__(
            premap_key=premap_key,
            postmap_key=postmap_key,
            parameter_map=parameter_map,
            behavior_map=behavior_map,
            mutator=parameter_map.mutate,
            equil_time=0,
        )
        self.periode = max(1, int(periode))
        self.knn = max(1, int(knn))
        self._current_goal: Optional[np.ndarray] = None
        self._current_goal_targeting_key = ""

    def suggest_trial(
        self,
        lookback_length: int = -1,
        goal: Optional[np.ndarray] = None,
        goal_targeting: Optional[Dict[str, Any]] = None,
    ) -> Any:
        feature_matrix = self._get_history_feature_matrix(lookback_length)

        if feature_matrix.shape[0] == 0:
            return self.parameter_map.sample()

        if goal is None:
            if self._should_refresh_goal(goal_targeting):
                if goal_targeting is None:
                    self._current_goal = self.behavior_map.sample(feature_matrix)
                else:
                    self._current_goal = self.behavior_map.sample(
                        feature_matrix,
                        goal_targeting=goal_targeting,
                    )
                self._current_goal_targeting_key = (
                    json.dumps(goal_targeting, sort_keys=True) if goal_targeting else ""
                )
            goal = self._current_goal

        if goal is None:
            if goal_targeting is None:
                goal = self.behavior_map.sample(feature_matrix)
            else:
                goal = self.behavior_map.sample(
                    feature_matrix,
                    goal_targeting=goal_targeting,
                )

        selected = self._nearest_params(
            goal=np.asarray(goal, dtype=float),
            lookback_length=lookback_length,
            k=self.knn,
        )
        base_policy = self._compose_base_policy(selected)
        return self.parameter_map.mutate(base_policy)

    def _should_refresh_goal(self, goal_targeting: Optional[Dict[str, Any]]) -> bool:
        if self._current_goal is None:
            return True
        goal_targeting_key = json.dumps(goal_targeting, sort_keys=True) if goal_targeting else ""
        if goal_targeting_key != self._current_goal_targeting_key:
            self._current_goal_targeting_key = goal_targeting_key
            return True
        return self.timestep % self.periode == 0

    def _distance_to_retrieval_history(
        self,
        goal: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        goal = np.asarray(goal, dtype=float).reshape(1, -1)
        min_ = features.min(axis=0)
        max_ = features.max(axis=0)
        denominator = max_ - min_
        denominator[denominator == 0] = 1.0
        return np.sum(((goal - features) / denominator) ** 2, axis=1)

    def _compose_base_policy(self, selected_params: List[Any]) -> Any:
        if not selected_params:
            return self.parameter_map.sample()

        if len(selected_params) == 1 or self.knn <= 1:
            return deepcopy(selected_params[0])

        return [deepcopy(params) for params in selected_params]


@expose
class BaseIMGEPExplorer(BaseExplorerFactory):
    config = BaseExplorerConfig
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, system: System) -> BaseIMGEPInstance:
        behavior_map = instantiate_object(
            self.config.behavior_map,
            system,
            object_name="behavior map",
        )
        param_map = instantiate_object(
            self.config.parameter_map,
            system,
            object_name="parameter map",
        )

        return BaseIMGEPInstance(
            parameter_map=param_map,
            behavior_map=behavior_map,
            periode=self.config.periode,
            knn=self.config.knn
        )
