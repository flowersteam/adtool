import dataclasses
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from typing import Dict

import numpy as np
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from examples.reku.systems.ReKu import ReKu, ReKuParams  # Adjust the import according to your project structure


class ReKuParameterMap(Leaf):
    def __init__(
        self,
        system: ReKu,

        premap_key: str = "params",
        param_obj: ReKuParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        self.perturbators = system.perturbators
        self.SCALING_FACTOR = 2*np.pi  # Define the scaling factor as needed
        self.locator = BlobLocator()
        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        # Uniform parameter maps for angular speeds and phases
        self.uniform_omega = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}_omega",
            tensor_low=np.full(self.perturbators, -  np.pi),
            tensor_high=np.full(self.perturbators,  np.pi),
            tensor_bound_low=np.full(self.perturbators, - np.pi),
            tensor_bound_high=np.full(self.perturbators,  np.pi)
        )

        self.uniform_phases = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}_phases",
            tensor_low=np.zeros(self.perturbators),
            tensor_high= np.ones(self.perturbators), 
            tensor_bound_low=np.zeros(self.perturbators),
            tensor_bound_high= np.ones(self.perturbators)
        )

        self.uniform_mutator_omega = partial(
            add_gaussian_noise,
            mean=np.zeros(self.perturbators),
            std=np.ones(self.perturbators) * 0.1
        )

        self.uniform_mutator_phases = partial(
            add_gaussian_noise,
            mean=np.zeros(self.perturbators),
            std=np.ones(self.perturbators) * 0.1
        )

        self.uniform_coupling_factor = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}_coupling_factor",
            tensor_low=np.array([0.1]),
            tensor_high=np.array([1]),
            tensor_bound_low=np.array([0.1]),
            tensor_bound_high=np.array([1])
        )

        self.uniform_mutator_coupling_factor = partial(
            add_gaussian_noise,
            mean=np.zeros(1),
            std=np.ones(1) * 0.1
        )


        

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        return intermed_dict

    def sample(self) -> Dict:
        pre_omega = self.uniform_omega.sample()
        pre_phases = self.uniform_phases.sample()


        phases = self.transform_parameters(pre_phases)

  

        p_dict = {
            "dynamic_params": asdict(ReKuParams(
                omega=pre_omega,
                initial_phases=phases,
                coupling_factor=1 #np.random.uniform(0.1, 1)

            ))
        }
        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)


        pre_omega =intermed_dict["dynamic_params"]['omega']
        pre_phases =intermed_dict["dynamic_params"]['initial_phases']

        mutated_omega = self.uniform_mutator_omega(pre_omega)
        mutated_phases_tensor = self.uniform_mutator_phases(pre_phases)
        mutated_phases_tensor = np.clip(mutated_phases_tensor, 0, 1)


        # coupling_factor =intermed_dict["dynamic_params"]['coupling_factor']
        # coupling_factor += np.random.normal(0, 0.1)
        # coupling_factor = np.clip(coupling_factor, 0.1, 1)


        mutated_phases = self.transform_parameters(mutated_phases_tensor)

        intermed_dict["dynamic_params"] = asdict(
            ReKuParams(
                omega=mutated_omega,
                initial_phases=mutated_phases,
                coupling_factor=1#coupling_factor
                
            )
        )


        return intermed_dict

    def transform_parameters(self, pre_params):
        for i in range(1, pre_params.size):
            pre_params[i] *= pre_params[i - 1]


        params = pre_params * self.SCALING_FACTOR
        return params

    def inverse_transform_parameters(self, params):
        pre_params = np.ones_like(params) / self.SCALING_FACTOR
        for i in range(1, params.size):
            pre_params[i] = params[i] / params[i - 1]
        return pre_params
