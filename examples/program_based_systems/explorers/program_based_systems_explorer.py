import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from adtool.explorers.IMGEPExplorer import IMGEPExplorerInstance
from adtool.mutators import SpecificMutator
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import ObjectSpec, instantiate_object, object_spec
from adtool.systems import System
import numpy as np

from adtool.utils.leaf.Leaf import Leaf
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
            "examples.program_based_systems.examples.core_interferences.behavior_map.InterferenceBehaviorMap.InterferenceBehaviorMap"
        )
    )
    parameter_map: ObjectSpec = Field(
        object_spec(
            "examples.program_based_systems.examples.core_interferences.parameter_map.InterferenceParameterMap.InterferenceParameterMap"
        )
    )
    history_optimization: ObjectSpec | None = None

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
        history_optimization: Any = None,
    ) -> None:
        if parameter_map is None or behavior_map is None:
            raise ValueError("BaseIMGEPInstance requires parameter_map and behavior_map.")
        super().__init__(
            premap_key=premap_key,
            postmap_key=postmap_key,
            parameter_map=parameter_map,
            behavior_map=behavior_map,
            mutator=SpecificMutator(),
            equil_time=0,
            history_optimization=history_optimization,
        )
        self.periode = max(1, int(periode))
        self.knn = max(1, int(knn))
        self._current_goal: Optional[np.ndarray] = None
        self._current_goal_targeting_key = ""

    def configure_experiment_runtime(self, *, config: Dict[str, Any], system: Any) -> bool:
        """Apply active program-map settings after checkpoint restoration.

        Checkpoints retain exploration history and runtime state, but the
        active config remains authoritative for stateless parameter and
        behavior policies.
        """
        explorer_config = config.get("explorer", {}).get("config", {})
        behavior_map_spec = explorer_config.get("behavior_map")
        if behavior_map_spec is not None:
            self.behavior_map = instantiate_object(
                behavior_map_spec,
                system,
                object_name="behavior map",
            )
            self._current_goal = None
            self._current_goal_targeting_key = ""
        parameter_map_spec = explorer_config.get("parameter_map")
        if parameter_map_spec is not None:
            self.parameter_map = instantiate_object(
                parameter_map_spec,
                system,
                object_name="parameter map",
            )
        self.periode = max(1, int(explorer_config.get("periode", self.periode)))
        self.knn = max(1, int(explorer_config.get("knn", self.knn)))
        optimization_spec = explorer_config.get("history_optimization")
        optimization = (
            instantiate_object(optimization_spec, object_name="history optimization")
            if optimization_spec is not None
            else None
        )
        self.history.set_history_optimization(optimization)
        return (
            behavior_map_spec is not None
            or parameter_map_spec is not None
            or optimization_spec is not None
        )

    def suggest_trial(
        self,
        history_lookback_length: int = -1,
        goal: Optional[np.ndarray] = None,
        goal_targeting: Optional[Dict[str, Any]] = None,
    ) -> Any:
        bounds = self.history.feature_bounds(history_lookback_length)
        if bounds is None:
            return self.parameter_map.sample()

        if goal is None:
            if self._should_refresh_goal(goal_targeting):
                if goal_targeting is None:
                    self._current_goal = self.behavior_map.sample_from_bounds(bounds)
                else:
                    self._current_goal = self.behavior_map.sample_from_bounds(
                        bounds,
                        goal_targeting=goal_targeting,
                    )
                self._current_goal_targeting_key = (
                    json.dumps(goal_targeting, sort_keys=True) if goal_targeting else ""
                )
            goal = self._current_goal

        if goal is None:
            if goal_targeting is None:
                goal = self.behavior_map.sample_from_bounds(bounds)
            else:
                goal = self.behavior_map.sample_from_bounds(
                    bounds,
                    goal_targeting=goal_targeting,
                )

        selected = self.history.nearest(
            np.asarray(goal, dtype=float),
            k=self.knn,
            history_lookback_length=history_lookback_length,
            normalized=True,
            normalization_bounds=bounds,
        )
        base_policy = self._compose_base_policy([match.payload for match in selected])
        return self.mutator(base_policy, parameter_map=self.parameter_map)

    def _should_refresh_goal(self, goal_targeting: Optional[Dict[str, Any]]) -> bool:
        if self._current_goal is None:
            return True
        goal_targeting_key = json.dumps(goal_targeting, sort_keys=True) if goal_targeting else ""
        if goal_targeting_key != self._current_goal_targeting_key:
            self._current_goal_targeting_key = goal_targeting_key
            return True
        return self.timestep % self.periode == 0

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
        history_optimization = (
            instantiate_object(
                self.config.history_optimization,
                object_name="history optimization",
            )
            if self.config.history_optimization is not None
            else None
        )

        return BaseIMGEPInstance(
            parameter_map=param_map,
            behavior_map=behavior_map,
            periode=self.config.periode,
            knn=self.config.knn,
            history_optimization=history_optimization,
        )
