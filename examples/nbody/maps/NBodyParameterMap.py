import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, Optional

import numpy as np
from adtool.systems import System
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf

from examples.nbody.systems.NBody import NBodyParams, NBodySimulation

class NBodyParameterMap(Leaf):
    def __init__(
        self,
        system: NBodySimulation,
        premap_key: str = "params",
        param_obj: NBodyParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = NBodyParams(
                speeds=np.zeros((system.N, 2)),
                positions=np.zeros((system.N, 2))
            )

        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        self.uniform = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}",
            tensor_low=np.full((system.N * 4,), -1).astype(np.float32),
            tensor_high=np.full((system.N * 4,), 1).astype(np.float32),
            tensor_bound_high=np.full((system.N * 4,), 1).astype(np.float32),
            tensor_bound_low=np.full((system.N * 4,), -1).astype(np.float32),
        )

        self.uniform_mutator = partial(
            add_gaussian_noise,
            mean=np.zeros(system.N * 4),
            std=np.full((system.N * 4,), 0.1),
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
        N = len(p_dyn_tensor) // 4
        dp = NBodyParams(
            speeds=p_dyn_tensor[:N*2].reshape(N, 2),
            positions=p_dyn_tensor[N*2:].reshape(N, 2)
        )
        p_dict = {
            "speeds": dp.speeds,
            "positions": dp.positions,
        }
        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)
        dp = NBodyParams(**parameter_dict)
 
        dp_tensor = np.concatenate([dp.speeds.flatten(), dp.positions.flatten()])
        mutated_dp_tensor = self.uniform_mutator(dp_tensor)
        N = len(mutated_dp_tensor) // 4
        intermed_dict["speeds"] = mutated_dp_tensor[:N*2].reshape(N, 2)
        intermed_dict["positions"] = mutated_dp_tensor[N*2:].reshape(N, 2)
        return intermed_dict