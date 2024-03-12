"""The simplest possible algorithm of Intrinsically Motivated Goal Exploration Processes
"""
from functools import partial
from typing import Any, Dict, List

import torch
from adtool_default.maps.LeniaParameterMap import LeniaParameterMap
from adtool_default.maps.LeniaStatistics import LeniaStatistics
from adtool_default.maps.MeanBehaviorMap import MeanBehaviorMap
from adtool_default.maps.UniformParameterMap import UniformParameterMap
from auto_disc.auto_disc.wrappers.IdentityWrapper import IdentityWrapper
from auto_disc.auto_disc.wrappers.mutators import add_gaussian_noise, call_mutate_method
from auto_disc.auto_disc.wrappers.SaveWrapper import SaveWrapper
from auto_disc.legacy.utils.config_parameters import (
    DictConfigParameter,
    IntegerConfigParameter,
    StringConfigParameter,
)
from auto_disc.utils.leaf.Leaf import Leaf
from auto_disc.utils.leaf.locators.locators import BlobLocator


@IntegerConfigParameter("equil_time", default=1, min=1)
@StringConfigParameter(
    "behavior_map", possible_values=["Mean", "LeniaStatistics"], default="Mean"
)
@DictConfigParameter("behavior_map_config", default={})
@StringConfigParameter(
    "parameter_map", possible_values=["Uniform", "LeniaParameterMap"], default="Uniform"
)
@DictConfigParameter("parameter_map_config", default={})
@StringConfigParameter(
    "mutator", possible_values=["gaussian", "specific"], default="specific"
)
@DictConfigParameter("mutator_config", default={})
class IMGEPFactory:
    """Factory class providing interface with config parameters and therefore the
    frontend
    """

    CONFIG_DEFINITION = {}

    # create specification for discovery attributes
    # TODO: kind of hard-coded for now, based on constructor defaults
    discovery_spec = ["params", "output", "raw_output", "rendered_output"]

    def __init__(self):
        # never called
        pass

    def __call__(self) -> "IMGEPExplorer":
        behavior_map = self.make_behavior_map()
        param_map = self.make_parameter_map()
        mutator = self.make_mutator(param_map)
        equil_time = self.config["equil_time"]
        explorer = IMGEPExplorer(
            parameter_map=param_map,
            behavior_map=behavior_map,
            equil_time=equil_time,
            mutator=mutator,
        )

        return explorer

    def make_behavior_map(self):
        kwargs = self.config["behavior_map_config"]
        if self.config["behavior_map"] == "Mean":
            behavior_map = MeanBehaviorMap(**kwargs)
        elif self.config["behavior_map"] == "LeniaStatistics":
            behavior_map = LeniaStatistics(**kwargs)
        else:
            # this branch should be unreachable,
            # because the ConfigParameter decorator checks
            raise Exception("unreachable")

        return behavior_map

    def make_parameter_map(self):
        kwargs = self.config["parameter_map_config"]
        if self.config["parameter_map"] == "Uniform":
            param_map = UniformParameterMap(**kwargs)
        elif self.config["parameter_map"] == "LeniaParameterMap":
            param_map = LeniaParameterMap(**kwargs)
        else:
            # this branch should be unreachable,
            # because the ConfigParameter decorator checks
            raise Exception("unreachable")

        return param_map

    def make_mutator(self, param_map: Any = None):
        if self.config["mutator"] == "specific":
            mutator = partial(call_mutate_method, param_map=param_map)
        elif self.config["mutator"] == "gaussian":
            mutator = partial(
                add_gaussian_noise, std=self.config["mutator_config"]["std"]
            )
        else:
            mutator = torch.nn.Identity()

        return mutator


class IMGEPExplorer(Leaf):
    """Basic IMGEP that diffuses in goalspace.

    A class instance of `IMGEPExplorer` has access to a provided
    `parameter_map`, `behavior_map`, and `mutator` as attributes, whereas it
    receives data from the system under study through the `.map` method.
    """

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "params",
        parameter_map: Leaf = IdentityWrapper(),
        behavior_map: Leaf = IdentityWrapper(),
        mutator: Leaf = torch.nn.Identity(),
        equil_time: int = 0,
    ) -> None:
        super().__init__()
        self.locator = BlobLocator()
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
        data_dict[self.postmap_key] = params_init

        # first timestep recorded
        # NOTE: therefore, regardless of self.equil_time, 1 equil step
        # will always happen
        data_dict["equil"] = 1
        self.timestep += 1

        return data_dict

    def map(self, system_output: Dict) -> Dict:
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
        new_trial_data = self.observe_results(system_output)

        # save results
        trial_data_reset = self._history_saver.map(new_trial_data)

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
            params_trial = self.suggest_trial()

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

    def suggest_trial(self, lookback_length: int = -1) -> torch.Tensor:
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
        goal = self.behavior_map.sample()

        source_policy = self._vector_search_for_goal(goal, lookback_length)

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
    ) -> torch.Tensor:
        """Extract tensor history from an array of dicts with labelled data,
        with the tensor being labelled by key.
        """
        # append history of tensors along a new dimension at index 0
        tensor_history = dict_history[0][key].unsqueeze(0)
        for dict in dict_history[1:]:
            tensor_history = torch.cat((tensor_history, dict[key].unsqueeze(0)), dim=0)

        return tensor_history

    def _find_closest(self, goal: torch.Tensor, goal_history: torch.Tensor):
        # TODO: simple L2 distance right now
        return torch.argmin((goal_history - goal).pow(2).sum(-1))

    def _vector_search_for_goal(self, goal: torch.Tensor, lookback_length: int) -> Dict:
        history_buffer = self._history_saver.get_history(
            lookback_length=lookback_length
        )

        goal_history = self._extract_tensor_history(history_buffer, self.premap_key)
        source_policy_idx = self._find_closest(goal, goal_history)

        param_history = self._extract_dict_history(history_buffer, self.postmap_key)
        source_policy = param_history[source_policy_idx]

        return source_policy
