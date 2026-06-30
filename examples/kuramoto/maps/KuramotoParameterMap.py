import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict

import torch
import numpy as np
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from examples.kuramoto.systems.Kuramoto import Kuramoto, KuramotoParams



class KuramotoParameterMap(Leaf):
    def __init__(
        self,
        system: Kuramoto,

        premap_key: str = "params",
        param_obj: KuramotoParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        self.N = system.N
        self.K = system.K
        # 2 trop fort

        self.SCALING_FACTOR = 0.1



        self.locator = BlobLocator()
        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        self.uniform_intra = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}_intra",
            tensor_low=np.zeros(self.N),
            tensor_high=np.ones(self.N),
            tensor_bound_low=np.zeros(self.N),
            tensor_bound_high=np.ones(self.N)
        )

        self.uniform_inter = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}_inter",
            tensor_low=np.zeros((self.N, self.N)),
            tensor_high=np.ones((self.N, self.N)),
            tensor_bound_low=np.zeros((self.N, self.N)),
            tensor_bound_high=np.ones((self.N, self.N))
        )

        self.uniform_mutator_intra = partial(
            add_gaussian_noise,
            mean=np.zeros(self.N),
            std=np.ones(self.N) * 0.1
        )

        self.uniform_mutator_inter = partial(
            add_gaussian_noise,
            mean=np.zeros((self.N, self.N)),
            std=np.ones((self.N, self.N)) * 0.1
        )

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        return intermed_dict

    def sample(self) -> Dict:
        pre_params_intra = self.uniform_intra.sample()
        inter_couplings = self.uniform_inter.sample()

        intra_couplings = self.transform_parameters(pre_params_intra)

        # Generate natural frequencies based on specified boundaries

        p_dict = {
            "dynamic_params": asdict(KuramotoParams(
                intra_couplings=intra_couplings,
                inter_couplings=inter_couplings,
            ))

            

        }
        p_dict["dynamic_params"]["pre_params_intra"] = pre_params_intra
        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)

        
        pre_params_intra = parameter_dict["dynamic_params"]["pre_params_intra"]
        pre_params_inter = parameter_dict["dynamic_params"]["inter_couplings"]


        mutated_intra_tensor = self.uniform_mutator_intra(pre_params_intra)

        mutated_inter_tensor = self.uniform_mutator_inter(pre_params_inter)

        mutated_intra_couplings = self.transform_parameters(mutated_intra_tensor)

        intermed_dict["dynamic_params"] = asdict(
            KuramotoParams(
                intra_couplings=mutated_intra_couplings,
                inter_couplings=mutated_inter_tensor,
            )
        )

        intermed_dict["dynamic_params"]["pre_params_intra"] = mutated_intra_tensor

        # Regenerate natural frequencies if needed (or keep the same if they should remain unchanged)

        return intermed_dict
    
    

    def transform_parameters(self, pre_params):
        params = np.ones_like(pre_params) * self.SCALING_FACTOR

        for i in range(1, pre_params.size):
            params[i] = params[i - 1] * pre_params[i]
        return params

    def inverse_transform_parameters(self, params):
        pre_params = np.ones_like(params) / self.SCALING_FACTOR
        for i in range(1, params.size):
            pre_params[i] = params[i] / params[i - 1]
        return pre_params
