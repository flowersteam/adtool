from functools import partial
from typing import Any, Dict
from adtool.explorers.IMGEPExplorer import IMGEPExplorerInstance as BaseIMGEPExplorerInstance
from adtool.systems import System
from adtool.wrappers.IdentityWrapper import IdentityWrapper
from adtool.wrappers.mutators import add_gaussian_noise, call_mutate_method
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import ObjectSpec, instantiate_object, object_spec
from adtool.utils.leaf.Leaf import Leaf
from pydantic import Field, BaseModel
import numpy as np
from enum import Enum
from scipy.spatial import KDTree

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
    novelty_weight: float = Field(0.5, ge=0, le=1)

class CuriosityDrivenIMGEP(BaseIMGEPExplorerInstance):
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
        super().__init__(
            premap_key=premap_key,
            postmap_key=postmap_key,
            parameter_map=parameter_map,
            behavior_map=behavior_map,
            mutator=mutator,
            equil_time=equil_time,
        )
        self.uncertainty_map = None
        self.kdtree = None
        self.novelty_weight = novelty_weight
        self._retrieval_features = np.zeros((0, 0), dtype=float)

    def update_uncertainty_map(self):
        goals = self._get_history_feature_matrix(-1)
        self._retrieval_features = goals

        if len(goals) == 0:
            self.kdtree = None
            self.uncertainty_map = None
            return

        self.kdtree = KDTree(goals)
        
        k = min(len(goals), 10)
        distances, indices = self.kdtree.query(goals, k=k)
        if len(distances.shape) == 1:
            distances = np.expand_dims(distances, axis=1)

        self.uncertainty_map = np.mean(distances, axis=1)

    def sample_curious_goal(self):
        if (
            self.uncertainty_map is None
            or self.kdtree is None
            or self._retrieval_features.shape[0] == 0
            or np.random.rand() < 0.1
        ):
            return self.behavior_map.sample()
        
        probs = self.uncertainty_map / np.sum(self.uncertainty_map)
        idx = np.random.choice(len(probs), p=probs)
        return self._retrieval_features[idx]
    
    def suggest_trial(
        self,
        lookback_length: int = -1,
        goal: np.ndarray = None,
        goal_targeting: Dict[str, Any] | None = None,
    ):
        self.update_uncertainty_map()
        
        if goal is None:
            if self.kdtree is None or np.random.rand() < self.novelty_weight:
                goal = self.behavior_map.sample()
            else:
                goal = self.sample_curious_goal()

        source_policy = self._vector_search_for_goal(goal, lookback_length)
        params_trial = self.mutator(source_policy)
        return params_trial

    def _vector_search_for_goal(self, goal: np.ndarray, lookback_length: int) -> Dict:
        selected = self._nearest_params(np.asarray(goal, dtype=float), lookback_length, k=1)
        if not selected:
            return self.parameter_map.sample()
        return selected[0]

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
        if self.config.mutator == MutatorEnum.Specific:
            mutator = partial(call_mutate_method, param_map=param_map)
        elif self.config.mutator == MutatorEnum.Gaussian:
            mutator = partial(
                add_gaussian_noise, std=self.config.mutator_config["std"]
            )
        else:
            mutator = None
        return mutator
