import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, Optional

import torch
import numpy as np
from adtool.systems import System
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator

from examples.block_ca.systems.Block import Block  



@dataclass
class BlockParams:
    a: float = 0.0  # Parameter a
    b: float = 0.0  # Parameter b
    c: float = 0.0  # Parameter c
    d: float = 0.0  # Parameter d
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    s: float = 0.0


    def to_tensor(self):
        return torch.tensor([self.a, self.b, self.c, self.d, self.p, self.q, self.r, self.s], dtype=torch.float32)
    
    def to_numpy(self):
        return np.array([self.a, self.b, self.c, self.d, self.p, self.q, self.r, self.s])

    @classmethod
    def from_tensor(cls, tensor):
        return cls(
            a=tensor[0].item(), b=tensor[1].item(), c=tensor[2].item(),
            d=tensor[3].item(), p=tensor[4].item(), q=tensor[5].item(),
            r=tensor[6].item(), s=tensor[7].item()
        )

class BlockParameterMap(Leaf):
    def __init__(
        self,
        system: Block,
        premap_key: str = "params",
        param_obj: BlockParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = BlockParams(a=0.0, b=0.0, c=0.0, d=0.0, p=0.0, q=0.0, r=0.0, s=0.0)

        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        self.uniform = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}",
            tensor_low=torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=torch.float32).numpy(),
            tensor_high=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).numpy(),
        )

        self.uniform_mutator = partial(
            add_gaussian_noise,
            mean=param_obj.to_tensor().numpy(),
            std=torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32).numpy(),
        )

    def sample(self) -> Dict:
        p_dyn_tensor = self.uniform.sample()

        p_dyn_tensor = self.normalize(p_dyn_tensor)




        dp = BlockParams.from_tensor(p_dyn_tensor)
        p_dict = {
            "dynamic_params": asdict(dp),
        }
        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        """
        Takes the dictionary of only parameters outputted by
        the explorer and mutates them.
        """
        intermed_dict = deepcopy(parameter_dict["dynamic_params"])


        # mutate dynamic parameters
        dp = BlockParams(**parameter_dict["dynamic_params"])
        dp_tensor = dp.to_numpy()
        mutated_dp_tensor = self.uniform_mutator(dp_tensor)

        mutated_dp_tensor = self.normalize(mutated_dp_tensor)
     #   print("BlockParams()",BlockParams())
        intermed_dict["dynamic_params"] = asdict(
            BlockParams().from_tensor(mutated_dp_tensor)
        )

     #   print("intermed_dict",intermed_dict['dynamic_params'])



        return intermed_dict
    
    def normalize(self, dp_tensor: torch.Tensor) -> torch.Tensor:
        a, b, c, d, p, q, r, s = dp_tensor
        norm = np.sqrt(a**2 + b**2 + c**2 + d**2)
        a, b, c, d = a / norm, b / norm, c / norm, d / norm
        norm = np.sqrt(p**2 + q**2 + r**2 + s**2)
        p, q, r, s = p / norm, q / norm, r / norm, s / norm
        return torch.tensor([a, b, c, d, p, q, r, s], dtype=torch.float32)


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