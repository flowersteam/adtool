"""The simplest possible algorithm of Intrinsically Motivated Goal Exploration Processes
"""
from functools import partial
import json
import os
from typing import Any, Dict, List, Union

from adtool.systems import System
from adtool.wrappers.IdentityWrapper import IdentityWrapper
from adtool.wrappers.mutators import add_gaussian_noise, call_mutate_method
from adtool.wrappers.SaveWrapper import SaveWrapper
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.leaf.Leaf import Leaf
from pydantic import Field
from pydoc import locate
from typing import Dict
from pydantic import BaseModel

import numpy as np

from enum import Enum

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

        if isinstance(system_output, list):
            new_trial_data = self.mean_var_goal(system_output)
        else:
            new_trial_data = self.observe_results(system_output)

        # save results
        trial_data_reset = self._history_saver.map( new_trial_data )


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
                goal=system_output['target'] if 'target' in system_output else None
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

    def suggest_trial(self, lookback_length: int = -1,
                      goal: np.ndarray = None
                      
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
            goal = self.behavior_map.sample()
       #     print("sampled goal", goal)

        source_policy = self._vector_search_for_goal(goal, lookback_length)
     #   source_policy = self._random_history_sample(lookback_length)

        # instead take a random policy

        params_trial = self.mutator(source_policy)


        return params_trial

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
        return self._history_saver.buffer[-1]

    def optimize(self):
        """Run optimization step for online learning of the `Explorer` policy."""
        pass

    def _extract_dict_history(self, dict_history: List[Dict], key: str) -> List[Dict]:
        """Extract history from an array of dicts with labelled data,
        with the desired subdict being labelled by key.
        """
        key_history = []
        for dict in dict_history:
            key_history.append(dict[key])
        return key_history

    def _extract_tensor_history(
        self, dict_history: List[Dict], key: str
    ) :
        """Extract tensor history from an array of dicts with labelled data,
        with the tensor being labelled by key.
        """
        # append history of tensors along a new dimension at index 0
        tensor_history = np.array([dict_history[0][key]])
        for dict in dict_history[1:]:
          #  tensor_history = torch.cat((tensor_history, dict[key].unsqueeze(0)), dim=0)
            tensor_history = np.concatenate((tensor_history, [dict[key]]), axis=0)

        return tensor_history

    def _find_closest(self, goal: np.ndarray,
                       goal_history: np.ndarray):
        # TODO: simple L2 distance right now
        # (200,17) , (17,)
        # return the argmin of the L2 distance with numpy

        return np.argmin(np.linalg.norm(goal_history - goal, axis=1))
    
    def _vector_search_for_goal(self, goal: np.ndarray, lookback_length: int) -> Dict:
        history_buffer = self._history_saver.get_history(
            lookback_length=lookback_length
        )


        goal_history = self._extract_tensor_history(history_buffer, self.premap_key)


        source_policy_idx = self._find_closest(goal, goal_history)


        param_history = self._extract_dict_history(history_buffer, self.postmap_key)
        source_policy = param_history[source_policy_idx]

        return source_policy

    def _random_history_sample(self, lookback_length: int) -> Dict:
        history_buffer = self._history_saver.get_history(
            lookback_length=lookback_length
        )

        source_policy_idx = np.random.randint(0, len(history_buffer))

        param_history = self._extract_dict_history(history_buffer, self.postmap_key)
        source_policy = param_history[source_policy_idx]

        return source_policy


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
        kwargs = self.config.behavior_map_config
        behavior_map=locate(self.config.behavior_map)(system,**kwargs)
        return behavior_map

    def make_parameter_map(self, system: System):
        kwargs = self.config.parameter_map_config
        param_map=locate(self.config.parameter_map)(system,**kwargs)
        return param_map

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
    



