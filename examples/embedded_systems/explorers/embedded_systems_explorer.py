from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from adtool.explorers.IMGEPExplorer import IMGEPExplorerInstance
from adtool.utils.expose_config.expose_config import expose
from adtool.systems import System
from examples.embedded_systems.helpers.module_factory import make_module
import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from examples.embedded_systems.behavior_map.embedded_systems_behavior_map import (
    BaseBehaviorMap,
)
from examples.embedded_systems.parameter_map.embedded_systems_parameter_map import (
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
    behavior_map_config: Dict = Field(default_factory=lambda: {
        "path": "examples.embedded_systems.examples.core_interferences.behavior_map.InterferenceBehaviorMap"
    })
    parameter_map_config: Dict = Field(default_factory=lambda: {
        "path": "examples.embedded_systems.examples.core_interferences.parameter_map.InterferenceParameterMap"
    })

class BaseIMGEPInstance(IMGEPExplorerInstance):
    """Embedded-systems IMGEP policy with periodic goals and kNN retrieval."""

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

    def suggest_trial(
        self,
        lookback_length: int = -1,
        goal: Optional[np.ndarray] = None,
        goal_targeting: Optional[Dict[str, Any]] = None,
    ) -> Any:
        feature_matrix, param_history = self._get_history_features(lookback_length)

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

        min_, max_ = self._compute_min_max(feature_matrix)
        indices = self._feature_to_closest_indices(
            goal=np.asarray(goal, dtype=float),
            features=feature_matrix,
            min_=min_,
            max_=max_,
        )

        selected = [param_history[i] for i in indices]
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

    def _get_history_features(self, lookback_length: int) -> Tuple[np.ndarray, List[Any]]:
        history_length = lookback_length
        if self._history_saver.locator.resource_uri == "":
            history_length = 1

        history = self._history_saver.get_history(lookback_length=history_length)
        feature_history = []
        param_history = []
        for item in history:
            feature = np.asarray(item.get(self.premap_key, []), dtype=float).reshape(-1)
            params = item.get(self.postmap_key, None)
            if params is None or feature.size == 0:
                continue
            if np.isnan(feature).any() or np.isinf(feature).any():
                continue
            feature_history.append(feature)
            param_history.append(params)

        if lookback_length > 0:
            feature_history = feature_history[-lookback_length:]
            param_history = param_history[-lookback_length:]

        if not feature_history:
            return np.zeros((0, 0), dtype=float), []

        feature_matrix = np.vstack(feature_history)
        return feature_matrix, param_history

    def _compute_min_max(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return features.min(axis=0), features.max(axis=0)

    def _feature_to_closest_indices(
        self,
        goal: np.ndarray,
        features: np.ndarray,
        min_: np.ndarray,
        max_: np.ndarray,
    ) -> np.ndarray:
        goal = goal.reshape(1, -1)
        denominator = max_ - min_
        denominator[denominator == 0] = 1.0
        distances = np.sum(((goal - features) / denominator) ** 2, axis=1)
        k_eff = min(self.knn, len(distances))
        return np.argsort(distances)[:k_eff]

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
        behavior_map = make_module(
            "behavior_map", system, **self.config.behavior_map_config)
        param_map = make_module("parameter_map", system,
                                **self.config.parameter_map_config)

        return BaseIMGEPInstance(
            parameter_map=param_map,
            behavior_map=behavior_map,
            periode=self.config.periode,
            knn=self.config.knn
        )
