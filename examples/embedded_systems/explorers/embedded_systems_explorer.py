from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from adtool.utils.expose_config.expose_config import expose
from build.lib.adtool.systems import System
from examples.embedded_systems.helpers.module_factory import make_module
import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from adtool.wrappers.SaveWrapper import SaveWrapper
from examples.embedded_systems.maps.embedded_systems_behavior_map import BaseBehaviorMap
from examples.embedded_systems.maps.embedded_systems_parameter_map import BaseParameterMap
from examples.embedded_systems.mixers.embedded_systems_mixer import BaseMixer


class BaseExplorerFactory(ABC):
    """Factory interface that builds a runnable explorer instance."""

    discovery_spec: List[str] = []

    @abstractmethod
    def __call__(self, system: Any) -> Leaf:
        ...

class BaseExplorerConfig(BaseModel):
    periode: int = Field(1, ge=1, le=100000)
    k: int = Field(1, ge=1, le=1000)
    behavior_map_config: Dict = Field(default_factory=lambda: {
        "path": "examples.embedded_systems.examples.core_interferences.maps.InterferenceBehaviorMap.InterferenceBehaviorMap"
    })
    parameter_map_config: Dict = Field(default_factory=lambda: {
        "path": "examples.embedded_systems.examples.core_interferences.maps.InterferenceParameterMap.InterferenceParameterMap"
    })

class BaseIMGEPInstance(Leaf):
    """Reusable IMGEP policy with kNN+mix behavior."""

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "params",
        parameter_map: BaseParameterMap | None = None,
        behavior_map: BaseBehaviorMap | None = None,
        periode: int = 1,
        k: int = 1,
        mixer: Optional[BaseMixer] = None,
    ) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        if parameter_map is None or behavior_map is None:
            raise ValueError("BaseIMGEPInstance requires parameter_map and behavior_map.")
        self.parameter_map = parameter_map
        self.behavior_map = behavior_map
        self.periode = max(1, int(periode))
        self.k = max(1, int(k))
        self.mixer = mixer

        self.timestep = 0
        self._history_saver = SaveWrapper()
        self._feature_cache: List[np.ndarray] = []
        self._param_cache: List[Any] = []
        self._history_cursor = 0
        self._current_goal: Optional[np.ndarray] = None

    def bootstrap(self) -> Dict[str, Any]:
        data_dict: Dict[str, Any] = {}
        data_dict[self.postmap_key] = self.parameter_map.sample()
        data_dict["equil"] = 1
        self.timestep += 1
        return data_dict

    def map(self, system_output: Dict) -> Dict:
        new_trial_data = self.observe_results(system_output)
        trial_data_reset = self._history_saver.map(new_trial_data)
        self._sync_history_cache()

        goal = system_output.get("target", None)
        params_trial = self.suggest_trial(goal=goal)
        trial_data_reset[self.postmap_key] = params_trial
        trial_data_reset = self.parameter_map.map(
            trial_data_reset, override_existing=False
        )
        trial_data_reset["equil"] = 0

        self.timestep += 1
        return trial_data_reset

    def observe_results(self, system_output: Dict) -> Dict:
        if system_output.get(self.premap_key, None) is not None:
            system_output = self.behavior_map.map(system_output)
        return system_output

    def read_last_discovery(self) -> Dict:
        return self._history_saver.buffer[-1]

    def suggest_trial(
        self,
        lookback_length: int = -1,
        goal: Optional[np.ndarray] = None,
    ) -> Any:
        feature_matrix, param_history = self._get_cached_history(lookback_length)

        if feature_matrix.shape[0] == 0:
            return self.parameter_map.sample()

        if goal is None:
            if self._should_refresh_goal():
                self._current_goal = self.behavior_map.sample()
            goal = self._current_goal

        if goal is None:
            goal = self.behavior_map.sample()

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

    def _should_refresh_goal(self) -> bool:
        if self._current_goal is None:
            return True
        return self.timestep % self.periode == 0

    def _sync_history_cache(self) -> None:
        buffer = self._history_saver.buffer
        if self._history_cursor > len(buffer):
            self._feature_cache = []
            self._param_cache = []
            self._history_cursor = 0

        while self._history_cursor < len(buffer):
            item = buffer[self._history_cursor]
            self._history_cursor += 1

            feature = np.asarray(item.get(self.premap_key, []), dtype=float).reshape(-1)
            params = item.get(self.postmap_key, None)
            if params is None or feature.size == 0:
                continue
            if np.isnan(feature).any() or np.isinf(feature).any():
                continue

            self._feature_cache.append(feature)
            self._param_cache.append(params)

    def _get_cached_history(self, lookback_length: int) -> Tuple[np.ndarray, List[Any]]:
        if lookback_length > 0:
            feature_history = self._feature_cache[-lookback_length:]
            param_history = self._param_cache[-lookback_length:]
        else:
            feature_history = self._feature_cache
            param_history = self._param_cache

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
        k_eff = min(self.k, len(distances))
        return np.argsort(distances)[:k_eff]

    def _compose_base_policy(self, selected_params: List[Any]) -> Any:
        if not selected_params:
            return self.parameter_map.sample()

        if len(selected_params) == 1 or self.k <= 1 or self.mixer is None:
            return deepcopy(selected_params[0])

        mixed_code = self._mix_codes(selected_params)
        return mixed_code

    def _mix_codes(self, selected_params: List[Any]) -> Any:
        if self.mixer is None:
            return deepcopy(selected_params[0])
        max_cycle = self._get_max_cycle()
        return self.mixer.mix(selected_params, max_cycle=max_cycle)

    def _get_max_cycle(self) -> int:
        param_obj = getattr(self.parameter_map, "param_obj", None)
        return int(getattr(param_obj, "max_cycle", 400))


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
            k=self.config.k
        )
