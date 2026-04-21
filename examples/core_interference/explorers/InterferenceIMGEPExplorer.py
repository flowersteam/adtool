from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from pydoc import locate

from adtool.systems import System
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.leaf.Leaf import Leaf
from adtool.wrappers.IdentityWrapper import IdentityWrapper
from adtool.wrappers.SaveWrapper import SaveWrapper
from examples.core_interference.helpers.modifiers.mix import mix_sequences
from examples.core_interference.helpers.modifiers.mix_interleaving import (
    mix_sequences_interleaved,
)
from examples.core_interference.helpers.modifiers.mix_preserving_time_strucuture import (
    mix_sequences as mix_sequences_preserv,
)


class InterferenceIMGEPConfig(BaseModel):
    # equil_time was removed from this explorer since it seems there was no use
    # to it, we instead kept and use bootstrap_size
    # Goal refresh period.
    periode: int = Field(1, ge=1, le=100000)
    # Number of nearest neighbors used by policy before optional program mixing.
    k: int = Field(1, ge=1, le=1000)
    # Number of chunks used by chunk/preserv mix operators.
    num_parts: int = Field(2, ge=1, le=64)
    # One of: chunks | preserv | interleaving.
    mix_type: str = Field("chunks")
    behavior_map: str = Field(
        "examples.core_interference.maps.InterferenceBehaviorMap.InterferenceBehaviorMap"
    )
    behavior_map_config: Dict = Field(default_factory=dict)
    parameter_map: str = Field(
        "examples.core_interference.maps.InterferenceParameterMap.InterferenceParameterMap"
    )
    parameter_map_config: Dict = Field(default_factory=dict)


class InterferenceIMGEPInstance(Leaf):
    """Interference-specific IMGEP policy with kNN+mix behavior."""

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "params",
        parameter_map: Leaf = IdentityWrapper(),
        behavior_map: Leaf = IdentityWrapper(),
        periode: int = 1,
        k: int = 1,
        num_parts: int = 2,
        mix_type: str = "chunks",
    ) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.parameter_map = parameter_map
        self.behavior_map = behavior_map
        self.periode = max(1, int(periode))
        self.k = max(1, int(k))
        self.num_parts = max(1, int(num_parts))
        self.mix_type = mix_type

        self.timestep = 0
        self._history_saver = SaveWrapper()
        # Keep a goal between iterations to have a `periode` behavior:
        # goals are not resampled at every step.
        self._current_goal: Optional[np.ndarray] = None

    def bootstrap(self) -> Dict:
        data_dict = {}
        # Random initialization: policy does not reuse
        # history yet and simply samples a fresh program pair.
        data_dict[self.postmap_key] = self.parameter_map.sample()
        data_dict["equil"] = 1
        self.timestep += 1
        return data_dict

    def map(self, system_output: Dict) -> Dict:
        # Convert raw system output into behavior-space observation and append it
        # to internal explorer history before selecting the next trial.
        new_trial_data = self.observe_results(system_output)
        trial_data_reset = self._history_saver.map(new_trial_data)

        # External targeting can override explorer-generated goals by placing
        # a `target` key in the pipeline payload.
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
        # behavior_map.map transforms simulator output into a fixed-size
        # vector under `self.premap_key`, and usually stores raw output as
        # `raw_output` for persistence/inspection.
        if system_output.get(self.premap_key, None) is not None:
            system_output = self.behavior_map.map(system_output)
        return system_output

    def read_last_discovery(self) -> Dict:
        return self._history_saver.buffer[-1]

    def suggest_trial(
        self,
        lookback_length: int = -1,
        goal: Optional[np.ndarray] = None,
    ) -> Dict:
        # History is filtered to remove malformed/NaN observations before policy
        # selection to avoid unstable distance computations.
        feature_matrix, param_history = self._get_valid_history(lookback_length)

        if feature_matrix.shape[0] == 0:
            return self.parameter_map.sample()

        if goal is None:
            if self._should_refresh_goal():
                # Goal sampling policy is delegated to the behavior map so this
                # explorer can stay domain-agnostic.
                self._current_goal = self.behavior_map.sample()
            goal = self._current_goal

        if goal is None:
            goal = self.behavior_map.sample()

        min_, max_ = self._compute_min_max(feature_matrix)
        # Policy uses a per-dimension normalization term (max-min) so
        # dimensions with larger scales do not dominate nearest-neighbor search.
        indices = self._feature_to_closest_indices(
            goal=np.asarray(goal, dtype=float),
            features=feature_matrix,
            min_=min_,
            max_=max_,
        )

        selected = [param_history[i] for i in indices]
        # Compose may mix multiple parents (k>1) before mutation
        base_policy = self._compose_base_policy(selected)
        return self.parameter_map.mutate(base_policy)

    def _should_refresh_goal(self) -> bool:
        if self._current_goal is None:
            return True
        return self.timestep % self.periode == 0

    def _get_valid_history(
        self, lookback_length: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        history_buffer = self._history_saver.get_history(lookback_length=lookback_length)

        feature_history = []
        param_history = []
        for item in history_buffer:
            feature = np.asarray(item.get(self.premap_key, []), dtype=float).reshape(-1)
            params = item.get(self.postmap_key, None)
            # Keep parameter and feature histories aligned by filtering both at
            # the same time. This avoids index mismatches during parent lookup.
            if params is None or feature.size == 0:
                continue
            if np.isnan(feature).any() or np.isinf(feature).any():
                continue
            feature_history.append(feature)
            param_history.append(params)

        if not feature_history:
            return np.zeros((0, 0), dtype=float), []

        feature_matrix = np.vstack(feature_history)
        return feature_matrix, param_history

    def _compute_min_max(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Per-dimension min/max for goal distance normalization.
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
        # Optimization policy behavior: neutralize constant
        # dimensions so division is always safe.
        denominator[denominator == 0] = 1.0
        distances = np.sum(((goal - features) / denominator) ** 2, axis=1)
        k_eff = min(self.k, len(distances))
        return np.argsort(distances)[:k_eff]

    def _compose_base_policy(self, selected_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not selected_params:
            return self.parameter_map.sample()

        if len(selected_params) == 1 or self.k <= 1:
            # Important: copy before mutate to avoid mutating objects retained
            # in history and accidentally corrupting replayed discoveries.
            return deepcopy(selected_params[0])

        core0_pool = [p["dynamic_params"]["core0"] for p in selected_params]
        core1_pool = [p["dynamic_params"]["core1"] for p in selected_params]

        mixed_core0, mixed_core1 = self._mix_cores(core0_pool, core1_pool)
        return {
            "dynamic_params": {
                "core0": mixed_core0,
                "core1": mixed_core1,
            }
        }

    def _mix_cores(
        self,
        core0_pool: List[Any],
        core1_pool: List[Any],
    ) -> Tuple[Any, Any]:
        # Read max_cycle from parameter map config to keep mixing aligned with
        # the same temporal constraints used by generation/mutation.
        param_obj = getattr(self.parameter_map, "param_obj", None)
        max_cycle = getattr(param_obj, "max_cycle", 400)

        if self.mix_type == "chunks":
            # Splits programs into temporal chunks and recombines them.
            core0 = mix_sequences(
                core0_pool, max_cycle=max_cycle, num_parts=self.num_parts
            )
            core1 = mix_sequences(
                core1_pool, max_cycle=max_cycle, num_parts=self.num_parts
            )
            return core0, core1

        if self.mix_type == "preserv":
            # Preserves temporal structure constraints while mixing chunks.
            core0 = mix_sequences_preserv(
                core0_pool, max_cycle=max_cycle, num_parts=self.num_parts
            )
            core1 = mix_sequences_preserv(
                core1_pool, max_cycle=max_cycle, num_parts=self.num_parts
            )
            return core0, core1

        if self.mix_type == "interleaving":
            # Interleaves instructions from candidate programs in time order.
            core0 = mix_sequences_interleaved(core0_pool, max_cycle=max_cycle)
            core1 = mix_sequences_interleaved(core1_pool, max_cycle=max_cycle)
            return core0, core1

        # Fallback to first candidate if a custom mix type is not recognized.
        return deepcopy(core0_pool[0]), deepcopy(core1_pool[0])


@expose
class InterferenceIMGEPExplorer:
    config = InterferenceIMGEPConfig
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, system: System) -> InterferenceIMGEPInstance:
        # Factory role: instantiate behavior/parameter maps from config paths so
        # this explorer can stay fully configuration-driven.
        behavior_map = self.make_behavior_map(system)
        param_map = self.make_parameter_map(system)

        return InterferenceIMGEPInstance(
            parameter_map=param_map,
            behavior_map=behavior_map,
            periode=self.config.periode,
            k=self.config.k,
            num_parts=self.config.num_parts,
            mix_type=self.config.mix_type,
        )

    def make_behavior_map(self, system: System):
        kwargs = self.config.behavior_map_config
        behavior_map_cls = locate(self.config.behavior_map)
        if behavior_map_cls is None:
            raise ValueError(
                f"Could not retrieve behavior map class from path: {self.config.behavior_map}."
            )
        return behavior_map_cls(system, **kwargs)

    def make_parameter_map(self, system: System):
        kwargs = self.config.parameter_map_config
        parameter_map_cls = locate(self.config.parameter_map)
        if parameter_map_cls is None:
            raise ValueError(
                f"Could not retrieve parameter map class from path: {self.config.parameter_map}."
            )
        return parameter_map_cls(system, **kwargs)
