from functools import partial
from typing import Any, Dict, List
from adtool.systems import System
from adtool.wrappers.IdentityWrapper import IdentityWrapper
from adtool.wrappers.mutators import add_gaussian_noise, call_mutate_method
from adtool.wrappers.SaveWrapper import SaveWrapper
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.leaf.Leaf import Leaf
from pydantic import Field, BaseModel
from pydoc import locate
import numpy as np
from enum import Enum
from scipy.spatial import KDTree

class MutatorEnum(Enum):
    Gaussian = 'gaussian'
    Specific = 'specific'

class IMGEPConfig(BaseModel):
    equil_time: int = Field(1, ge=1, le=1000)
    behavior_map: str = Field("adtool.maps.MeanBehaviorMap.MeanBehaviorMap")
    behavior_map_config: Dict = Field({})
    parameter_map: str = Field("adtool.maps.UniformParameterMap.UniformParameterMap")
    parameter_map_config: Dict = Field({})
    mutator: MutatorEnum = Field(MutatorEnum.Specific)
    mutator_config: Dict = Field({})
    novelty_weight: float = Field(0.5, ge=0, le=1)

class CuriosityDrivenIMGEP(Leaf):
    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "params",
        parameter_map: Leaf = IdentityWrapper(),
        behavior_map: Leaf = IdentityWrapper(),
        mutator: Leaf = Leaf(),
        equil_time: int = 0,
        novelty_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.parameter_map = parameter_map
        self.behavior_map = behavior_map
        self.equil_time = equil_time
        self.timestep = 0
        self.mutator = mutator
        self._history_saver = SaveWrapper()
        self.uncertainty_map = None
        self.kdtree = None
        self.novelty_weight = novelty_weight

    def bootstrap(self) -> Dict:
        data_dict = {}
        params_init = self.parameter_map.sample()
        data_dict[self.postmap_key] = params_init
        data_dict["equil"] = 1
        self.timestep += 1
        return data_dict

    def map(self, system_output: Dict) -> Dict:
        new_trial_data = self.observe_results(system_output)
        trial_data_reset = self._history_saver.map(new_trial_data)

        if self.timestep < self.equil_time:
            trial_data_reset = self.parameter_map.map(
                trial_data_reset, override_existing=True
            )
            trial_data_reset["equil"] = 1
        else:
            params_trial = self.suggest_trial(
                goal=system_output['target'] if 'target' in system_output else None
            )
            trial_data_reset[self.postmap_key] = params_trial
            trial_data_reset = self.parameter_map.map(
                trial_data_reset, override_existing=False
            )
            trial_data_reset["equil"] = 0

        self.timestep += 1
        return trial_data_reset

    def update_uncertainty_map(self):
        history = self._history_saver.get_history()
        goals = self._extract_tensor_history(history, self.premap_key)
        
        # Filter out NaN values
        valid_mask = ~np.isnan(goals).any(axis=1)
        valid_goals = goals[valid_mask]
        
        if len(valid_goals) == 0:
            self.kdtree = None
            self.uncertainty_map = None
            return

        self.kdtree = KDTree(valid_goals)
        
        k = min(len(valid_goals), 10)  # number of neighbors to consider 
        distances, indices = self.kdtree.query(valid_goals, k=k)
        if len(distances.shape) == 1:
            distances = np.expand_dims(distances, axis=1)

        self.uncertainty_map = np.mean(distances, axis=1)
        self.valid_indices = np.where(valid_mask)[0]
            


    def sample_curious_goal(self):
        if self.uncertainty_map is None or self.kdtree is None or \
        np.random.rand() < 0.1:  # Occasional random sampling
            return self.behavior_map.sample()
        
        probs = self.uncertainty_map / np.sum(self.uncertainty_map)
        idx = np.random.choice(len(probs), p=probs)
        return self.kdtree.data[idx]
    
    def suggest_trial(self, lookback_length: int = -1, goal: np.ndarray = None):
        self.update_uncertainty_map()
        
        if goal is None:
            if self.kdtree is None or np.random.rand() < self.novelty_weight:
                goal = self.behavior_map.sample()
            else:
                goal = self.sample_curious_goal()

        source_policy = self._vector_search_for_goal(goal, lookback_length)
        params_trial = self.mutator(source_policy)
        return params_trial

    def observe_results(self, system_output: Dict) -> Dict:
        if system_output.get(self.premap_key, None) is not None:
            system_output = self.behavior_map.map(system_output)
        return system_output

    def read_last_discovery(self) -> Dict:
        return self._history_saver.buffer[-1]

    def optimize(self):
        pass

    def _extract_dict_history(self, dict_history: List[Dict], key: str) -> List[Dict]:
        return [dict[key] for dict in dict_history]

    def _extract_tensor_history(self, dict_history: List[Dict], key: str):
        return np.array([dict[key] for dict in dict_history])

    def _find_closest(self, goal: np.ndarray, goal_history: np.ndarray):
        return np.argmin(np.linalg.norm(goal_history - goal, axis=1))

    def _vector_search_for_goal(self, goal: np.ndarray, lookback_length: int) -> Dict:
        history_buffer = self._history_saver.get_history(lookback_length=lookback_length)
        goal_history = self._extract_tensor_history(history_buffer, self.premap_key)
        
        # Filter out NaN values
        valid_mask = ~np.isnan(goal_history).any(axis=1)
        valid_goal_history = goal_history[valid_mask]
        
        if len(valid_goal_history) == 0:
            # If no valid goals, return a random policy
            return self.parameter_map.sample()
        
        source_policy_idx = self._find_closest(goal, valid_goal_history)
        param_history = self._extract_dict_history(history_buffer, self.postmap_key)
        valid_param_history = [param for i, param in enumerate(param_history) if valid_mask[i]]
        return valid_param_history[source_policy_idx]

@expose
class IMGEPExplorer():
    config = IMGEPConfig
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, system) -> CuriosityDrivenIMGEP:
        behavior_map = self.make_behavior_map(system)
        param_map = self.make_parameter_map(system)
        mutator = self.make_mutator(param_map)
        equil_time = self.config.equil_time
        novelty_weight = self.config.novelty_weight
        explorer = CuriosityDrivenIMGEP(
            parameter_map=param_map,
            behavior_map=behavior_map,
            equil_time=equil_time,
            mutator=mutator,
            novelty_weight=novelty_weight,
        )
        return explorer

    def make_behavior_map(self, system: System):
        kwargs = self.config.behavior_map_config
        behavior_map = locate(self.config.behavior_map)(system, **kwargs)
        return behavior_map

    def make_parameter_map(self, system: System):
        kwargs = self.config.parameter_map_config
        param_map = locate(self.config.parameter_map)(system, **kwargs)
        return param_map

    def make_mutator(self, param_map: Any = None):
        if self.config.mutator == MutatorEnum.Specific:
            mutator = partial(call_mutate_method, param_map=param_map)
        elif self.config.mutator == MutatorEnum.Gaussian:
            mutator = partial(
                add_gaussian_noise, std=self.config.mutator_config["std"]
            )
        else:
            mutator = None
        return mutator