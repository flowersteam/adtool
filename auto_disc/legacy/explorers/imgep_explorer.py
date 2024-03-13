import typing
from copy import deepcopy
from typing import Callable, Type

import numpy as np
import torch
from addict import Dict
from auto_disc.legacy.explorers import BaseExplorer
from auto_disc.legacy.utils.config_parameters import (
    BooleanConfigParameter,
    DecimalConfigParameter,
    IntegerConfigParameter,
    StringConfigParameter,
)
from auto_disc.legacy.utils.spaces import BoxSpace, DictSpace
from torch import Tensor, nn


# @StringConfigParameter(
#     name="source_policy_selection_type", possible_values=["optimal"], default="optimal"
# )
# @StringConfigParameter(
#     name="goal_selection_type",
#     possible_values=["random", "specific", "function", None],
#     default="random",
# )
# @IntegerConfigParameter(name="num_of_random_initialization", default=10, min=1)
# @BooleanConfigParameter(name="use_exandable_goal_space", default=True)

from auto_disc.auto_disc.utils.expose_config.defaults import Defaults, defaults
from dataclasses import dataclass, field

@dataclass
class IMGEPExplorerConfig(Defaults):
    source_policy_selection_type: str = defaults("optimal", domain=["optimal"])
    goal_selection_type: str = defaults("random", domain=["random", "specific", "function", None])
    num_of_random_initialization: int = defaults(10, min=1)
    use_exandable_goal_space: bool = defaults(True)

@IMGEPExplorerConfig.expose_config()
class IMGEPExplorer(BaseExplorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(
        self,
        input_space: DictSpace,
        output_space: DictSpace,
        input_distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Defines input and output space for the explorer (as well as a distance function for the input space).

        Args:
            input_space: space providing by the last output_representations
            output_space: space providing by the first input_wrappers
            input_distance_fn: a method to calc distance
        """
        super().initialize(input_space, output_space, input_distance_fn)
        if len(self.input_space) > 1:
            raise NotImplementedError("Only 1 vector can be accepted as input space")
        self._outter_input_space_key = list(self.input_space.spaces.keys())[
            0
        ]  # select first key in DictSpace

    def expand_box_goal_space(
        self, space: BoxSpace, observations: torch.Tensor
    ) -> None:
        """
        Expand the goal space in which the explorer will explore

        Args:
            space: current goal space
            observations: observations/results of previous explorations
        """
        observations = observations.type(space.dtype)
        is_nan_mask = torch.isnan(observations)
        if is_nan_mask.sum() > 0:
            observations[is_nan_mask] = space.low[is_nan_mask]
            observations[is_nan_mask] = space.high[is_nan_mask]
        space.low = torch.min(space.low, observations)
        space.high = torch.max(space.high, observations)

    def _get_next_goal(self) -> torch.Tensor:
        """Defines the next goal of the exploration."""

        if self.config.goal_selection_type == "random":
            target_goal = self.input_space.sample()
        else:
            raise ValueError(
                "Unknown goal generation type {!r} in the configuration!".format(
                    self.config.goal_selection_type
                )
            )

        return target_goal[self._outter_input_space_key]

    def _get_source_policy_idx(
        self, target_goal: torch.Tensor, history: list
    ) -> torch.Tensor:
        """
        get idx of source policy which should be mutated
        Args:
            target_goal: The aim goal
            history: previous observations

        Returns:
            source_policy_idx: idx int tensor
        """
        goal_library = torch.stack(
            [h[self._outter_input_space_key] for h in history]
        )  # get goal history as tensor

        if self.config.source_policy_selection_type == "optimal":
            # get distance to other goals
            goal_distances = self.input_distance_fn(target_goal, goal_library)

            # select goal with minimal distance
            source_policy_idx = torch.argmin(goal_distances)
        else:
            raise ValueError(
                "Unknown source policy selection type {!r} in the configuration!".format(
                    self.config.source_policy_selection_type
                )
            )

        return source_policy_idx

    def sample(self) -> Dict:
        """
        Emits a new set of parameters to test in the system
        """
        target_goal = None
        source_policy_idx = None
        policy_parameters = Dict()  # policy parameters (output of IMGEP policy)

        # random sampling if not enough in library
        if self.CURRENT_RUN_INDEX < self.config.num_of_random_initialization:
            # initialize the parameters
            policy_parameters = self.output_space.sample()
            # for parameter_key, parameter_space in self._output_space.items():
            #     policy_parameters[parameter_key] = sample_value(parameter_space)

        else:
            # sample a goal space from the goal space
            target_goal = self._get_next_goal()

            # get source policy which should be mutated
            history = self._access_history()
            source_policy_idx = self._get_source_policy_idx(
                target_goal, history["input"]
            )
            source_policy = history[int(source_policy_idx)]["output"]

            policy_parameters = self.output_space.mutate(source_policy)

        # TODO: Target goal
        # run with parameters
        # self._convert_policy_to_run_parameters(policy_parameters)
        run_parameters = deepcopy(policy_parameters)

        return run_parameters

    def observe(self, parameters: Dict, observations: typing.Dict[str, Tensor]) -> None:
        """
        gives to the explorer the exit of the system after using the parameters

        Args:
            parameters: system's parameters
            observations: observations/results of previous explorations
        """
        if self.config.use_exandable_goal_space:
            self.expand_box_goal_space(
                self.input_space[self._outter_input_space_key],
                observations[self._outter_input_space_key],
            )
            self.logger.debug("Imgep goal space was extended")

    def optimize(self):
        pass

    def save(self) -> typing.Dict[str, DictSpace]:
        return {"input_space": self.input_space}

    def load(self, saved_dict) -> None:
        self.input_space = saved_dict["input_space"]
