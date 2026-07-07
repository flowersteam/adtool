"""The simplest possible algorithm of Intrinsically Motivated Goal Exploration Processes
"""
from functools import partial
import json
import os
from typing import Any, Dict, List, Union
from adtool.utils.leaf.locators.locators import BlobLocator

from adtool.systems import System
from adtool.explorers.history_store import HistoryStore
from adtool.wrappers.IdentityWrapper import IdentityWrapper
from adtool.wrappers.mutators import add_gaussian_noise, call_mutate_method
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import ObjectSpec, instantiate_object, object_spec
from adtool.utils.leaf.Leaf import Leaf, prune_state
from pydantic import Field
from typing import Dict
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
    lookback_length: int = Field(-1, ge=-1)



class IMGEPExplorerInstance(Leaf):
    """Basic IMGEP that diffuses in goalspace.

    A class instance of `IMGEPExplorerInstance` has access to a provided
    `parameter_map`, `behavior_map`, and `mutator` as attributes, whereas it
    receives data from the system under study through the `.map` method.
    """

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "params",
        parameter_map: Leaf = IdentityWrapper(),
        behavior_map: Leaf = IdentityWrapper(),
        mutator: Leaf = Leaf(),
        equil_time: int = 0,
        lookback_length: int = -1,
    ) -> None:
        super().__init__()
        
        if lookback_length != 1:
            self.locator = BlobLocator()
        
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.parameter_map = parameter_map
        self.behavior_map = behavior_map
        self.equil_time = equil_time
        self.lookback_length = lookback_length
        self.timestep = 0

        self.mutator = mutator

        self._history_saver = HistoryStore(retain_full_history=False)
        self._history_saver.configure_retrieval_view(self._extract_retrieval_view)

    def bootstrap(self) -> Dict:
        """Return an initial sample needed to bootstrap the exploration loop."""
        data_dict = {}
        # initialize sample
        params_init = self.parameter_map.sample()

   #     params_init = np.concatenate((params_init, variances), axis=0)


        data_dict[self.postmap_key] = params_init


        # first timestep recorded
        # NOTE: therefore, regardless of self.equil_time, 1 equil step
        # will always happen
        data_dict["equil"] = 1
        self.timestep += 1


        return data_dict
    
    def mean_var_goal(self, system_output:List[Dict]) -> Dict:
        new_trial_data = []
        for sys_out in system_output:
            new_trial_data.append(self.observe_results(sys_out))
        
        var = np.var(
            np.array([trial_data["output"] for trial_data in new_trial_data]), axis=0
        )
        var=var/(var+1)
        # same but with a variance always between 0 and 1
        
        mean = np.mean(
            np.array([trial_data["output"] for trial_data in new_trial_data]), axis=0
        )
        new_trial_data=new_trial_data[0]
        new_trial_data["output"] =  np.concatenate((mean, var), axis=0)
        new_trial_data.pop("raw_output", None)
        return new_trial_data


    def map(self, system_output: 
            Union[Dict, List[Dict]]
            
            ) -> Dict:
        """Map the raw output of the system rollout to a subsequent parameter
        configuration to try.

        Args:
            system_output:
                A dictionary where the key `self.premap_key` indexes the raw
                output given at the end of the previous system rollout.

        Returns:
            A dictionary where the key `self.postmap_key` indexes the parameters to try in the next trial.
        """
        # either do nothing, or update dict by changing "output" -> "raw_output"
        # and adding new "output" key which is the result of the behavior map

        target, goal_targeting = self._extract_external_controls(system_output)

        if isinstance(system_output, list):
            new_trial_data = self.mean_var_goal(system_output)
        else:
            new_trial_data = self.observe_results(system_output)

        # save results
        trial_data_reset = self._record_discovery(new_trial_data)


        # TODO: check gradients here
        if self.timestep < self.equil_time:


            # sets "params" key
            trial_data_reset = self.parameter_map.map(
                trial_data_reset, override_existing=True
            )

            # label which trials were from random initialization
            trial_data_reset["equil"] = 1
        else:
            # suggest_trial reads history
            params_trial = self.suggest_trial(
                goal=target,
                goal_targeting=goal_targeting,
            )

            # assemble dict and update parameter_map state
            # NOTE: that this pass through parameter_map should not modify
            # the "params" data, but only so that parameter_map can update
            # its own state from reading the new parameters
            trial_data_reset[self.postmap_key] = params_trial
            trial_data_reset = self.parameter_map.map(
                trial_data_reset, override_existing=False
            )

            # label that trials are now from the usual IMGEP procedure
            trial_data_reset["equil"] = 0

        self.timestep += 1

        return trial_data_reset

    def suggest_trial(
        self,
        lookback_length: int = -1,
        goal: np.ndarray = None,
        goal_targeting: Dict[str, Any] | None = None,
    ):
        """Sample according to the policy a new trial of parameters for the
        system.

        Args:
            lookback_length:
                number of previous trials to consider when choosing the next
                trial, i.e., it is a batch size based on the save frequency.

                Note that the default `lookback_length = -1` will retrieve the
                entire  history.

        Returns:
            A `torch.Tensor` containing the parameters to try.
        """
        if goal is None:
            if goal_targeting is None:
                goal = self.behavior_map.sample()
            else:
                goal = self.behavior_map.sample(goal_targeting=goal_targeting)
       #     print("sampled goal", goal)

        source_policy = self._vector_search_for_goal(goal, lookback_length)
     #   source_policy = self._random_history_sample(lookback_length)

        # instead take a random policy

        params_trial = self.mutator(source_policy)


        return params_trial

    def _extract_external_controls(
        self,
        system_output: Union[Dict, List[Dict]],
    ) -> tuple[np.ndarray | None, Dict[str, Any] | None]:
        if isinstance(system_output, list):
            source = system_output[0] if system_output else {}
        else:
            source = system_output

        if not isinstance(source, dict):
            return None, None

        return source.get("target"), source.get("goal_targeting")

    def observe_results(self, system_output: Dict) -> Dict:
        """Read the raw output observed and process it into a discovered
        behavior.

        Args:
            system_output: See arguments for `.map`.

        Returns:
            A dictionary of the observed behavior/feature vector associated with
            the raw `system_output`
        """
        # check we are not in the initialization case
        if system_output.get(self.premap_key, None) is not None:
            # recall that behavior_maps will remove the dict entry of
            # self.premap_key
            system_output = self.behavior_map.map(system_output)
        else:
            pass

        return system_output

    def read_last_discovery(self) -> Dict:
        """Return last observed discovery."""
        return self._history_saver.last()

    def optimize(self):
        """Run optimization step for online learning of the `Explorer` policy."""
        pass

    def _extract_retrieval_view(
        self,
        item: Dict[str, Any],
    ) -> tuple[np.ndarray, Any] | None:
        """Build the retrieval-cache view of one stored discovery.

        By the time a discovery reaches history, `behavior_map.map(...)` has
        usually already converted raw system output into behavior space. This
        method does not redo that projection. It simply selects:

        - the numeric feature vector used to index nearest neighbors
        - the payload returned when a neighbor is selected
        """
        feature = np.asarray(item.get(self.premap_key, []), dtype=float).reshape(-1)
        params = item.get(self.postmap_key, None)
        if params is None or feature.size == 0:
            return None
        if np.isnan(feature).any() or np.isinf(feature).any():
            return None
        return feature, params

    def _record_discovery(self, discovery: Dict) -> Dict:
        """Store one discovery through the shared history interface."""
        return self._history_saver.record(discovery)

    def _get_history_feature_matrix(self, lookback_length: int) -> np.ndarray:
        return self._history_saver.get_retrieval_features(lookback_length=lookback_length)

    def _get_history_features(self, lookback_length: int) -> tuple[np.ndarray, List[Any]]:
        return self._history_saver.get_retrieval_view(lookback_length=lookback_length)

    def _distance_to_retrieval_history(
        self,
        goal: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        goal = np.asarray(goal, dtype=float).reshape(1, -1)
        return np.sum((goal - features) ** 2, axis=1)

    def _nearest_params(
        self,
        goal: np.ndarray,
        lookback_length: int,
        *,
        k: int = 1,
    ) -> List[Any]:
        return self._history_saver.find_nearest_payloads(
            goal,
            k=k,
            lookback_length=lookback_length,
            distance_fn=self._distance_to_retrieval_history,
        )

    def _nearest_indices(
        self,
        goal: np.ndarray,
        lookback_length: int,
        *,
        k: int = 1,
    ) -> np.ndarray:
        return self._history_saver.find_nearest_indices(
            goal,
            k=k,
            lookback_length=lookback_length,
            distance_fn=self._distance_to_retrieval_history,
        )
    
    def _vector_search_for_goal(self, goal: np.ndarray, lookback_length: int) -> Dict:
        selected = self._nearest_params(
            np.asarray(goal, dtype=float),
            lookback_length,
            k=1,
        )
        if not selected:
            return self.parameter_map.sample()
        return selected[0]

    def _random_history_sample(self, lookback_length: int) -> Dict:
        _, param_history = self._get_history_features(lookback_length)
        if not param_history:
            return self.parameter_map.sample()
        source_policy_idx = np.random.randint(0, len(param_history))
        return param_history[source_policy_idx]

    @prune_state({"_history_saver": None})
    def serialize(self) -> bytes:
        return super().serialize()


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
        lookback_length = self.config.lookback_length
        explorer = IMGEPExplorerInstance(
            parameter_map=param_map,
            behavior_map=behavior_map,
            equil_time=equil_time,
            mutator=mutator,
            lookback_length=lookback_length,
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
    
