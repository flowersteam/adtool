import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, Optional

from pydantic import BaseModel
import torch
import numpy as np
from adtool.systems import System
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator

from examples.block_ca.systems.Block import Block  


class DraftParams(BaseModel):
    # TODO global parameters that define all systems, use Field to define the acceptable range of values
    pass


@dataclass
class DraftDynamicParams:
    # TODO parameters that define a specific system


    def to_tensor(self):
        # TODO return a tensor representation of the parameters
        pass

    @classmethod
    def from_tensor(cls, tensor):
        # return a class instance from a tensor
        return cls(
            # TODO assign tensor values to class attributes
        )


class DraftParameterMap(Leaf):
    def __init__(
        self,
        system: Block,
        premap_key: str = "params",
        param_obj: DraftDynamicParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = DraftDynamicParams()

        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key




    def sample(self) -> Dict:
        # TODO sample a random set of parameters (for initialisation) as a dictionary

        return {
            "dynamic_params": {
                # TODO random parameters
            },
        }

    def mutate(self, parameter_dict: Dict) -> Dict:
        """
        Takes the dictionary of only parameters outputted by
        the explorer and mutates them.
        """
        intermed_dict = deepcopy(parameter_dict)

        # TODO: mutate dynamic_params in it

        return intermed_dict


    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        """
        Takes input dictionary and overrides the
        premap_key with generated parameters for Lenia.
        """
        intermed_dict = deepcopy(input)

        # check if either "params" is not set or if we want to override
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            # overrides "params" with new sample
            intermed_dict[self.premap_key] = self.sample()
        else:
            # passes "params" through if it exists
            pass

        return intermed_dict