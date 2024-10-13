import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, List

import torch
import numpy as np
from adtool.utils.leaf.Leaf import Leaf
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from examples.flashlenia.systems.FlashLenia import FlashLenia

@dataclass
class FlashLeniaParams:
    kernel: List[float]

    def to_tensor(self):
        return torch.tensor(self.kernel, dtype=torch.float32)
    
    def to_numpy(self):
        return np.array(self.kernel)

    @classmethod
    def from_tensor(cls, tensor):
        return cls(kernel=tensor.tolist())

class FlashLeniaParameterMap(Leaf):
    def __init__(
        self,
        system: FlashLenia,
        premap_key: str = "params",
        param_obj: FlashLeniaParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = FlashLeniaParams(kernel=[0.05, 0.2, 0.05, 0.2, 0.0, 0.2, 0.05, 0.2, 0.05])

        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        self.uniform = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}",
            tensor_low=torch.tensor([0.0] * 9, dtype=torch.float32).numpy(),
            tensor_high=torch.tensor([1.0] * 9, dtype=torch.float32).numpy(),
        )

        self.uniform_mutator = partial(
            add_gaussian_noise,
            mean=param_obj.to_tensor().numpy(),
            std=torch.tensor([0.05] * 9, dtype=torch.float32).numpy(),
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
        dp = FlashLeniaParams.from_tensor(p_dyn_tensor)
        p_dict = {
            "dynamic_params": asdict(dp),
        }
        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)
        dp = FlashLeniaParams(**parameter_dict["dynamic_params"])
        dp_tensor = dp.to_tensor()
        mutated_dp_tensor = self.uniform_mutator(dp_tensor)
        intermed_dict["dynamic_params"] = asdict(
            FlashLeniaParams.from_tensor(mutated_dp_tensor)
        )
        return intermed_dict