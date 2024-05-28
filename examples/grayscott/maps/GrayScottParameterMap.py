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


@dataclass
class GrayScottParams:
    F: float  # Feed rate
    k: float  # Kill rate
    Du: float  # Diffusion rate of U
    Dv: float  # Diffusion rate of V

    def to_tensor(self):
        return torch.tensor([self.F, self.k, self.Du, self.Dv], dtype=torch.float32)

    @classmethod
    def from_tensor(cls, tensor):
        return cls(F=tensor[0].item(), k=tensor[1].item(), Du=tensor[2].item(), Dv=tensor[3].item())


class GrayScottParameterMap(Leaf):
    def __init__(
        self,
        premap_key: str = "params",
        param_obj: GrayScottParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = GrayScottParams(F=0.04, k=0.06, Du=0.16, Dv=0.08)

        self.locator = BlobLocator()
        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        self.uniform = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}",
            tensor_low=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            tensor_high=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            tensor_bound_low=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            tensor_bound_high=torch.tensor([0.1, 0.1, 1.0, 1.0]),
        )

        self.uniform_mutator = partial(
            add_gaussian_noise,
            mean=param_obj.to_tensor(),
            std=torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=torch.float32)
        )

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        return intermed_dict

    def sample(self) -> Dict:
        p_dyn_tensor = self.uniform.sample()
        dp = GrayScottParams.from_tensor(p_dyn_tensor)
        p_dict = {
            "dynamic_params": asdict(dp),
        }
        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)
        dp = GrayScottParams(**parameter_dict["dynamic_params"])
        dp_tensor = dp.to_tensor()
        mutated_dp_tensor = self.uniform_mutator(dp_tensor)
        intermed_dict["dynamic_params"] = asdict(
            GrayScottParams.from_tensor(mutated_dp_tensor)
        )
        return intermed_dict
