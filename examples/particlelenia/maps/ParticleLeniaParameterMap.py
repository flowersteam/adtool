import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, Optional

import torch
from adtool.utils.misc.torch_utils import replace_torch_with_numpy
from examples.particlelenia.systems.ParticleLenia import ParticleLenia
from examples.particlelenia.systems.ParticleLeniaParameters import (
    ParticleLeniaDynamicalParameters,
    ParticleLeniaHyperParameters,
)
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator


class ParticleLeniaParameterMap(Leaf):
    """
    Due to the complexities of initializing ParticleLenia parameters,
    it's easier to make this custom parameter map.
    """

    def __init__(
        self,
        system: ParticleLenia,
        premap_key: str = "params",
        param_obj: ParticleLeniaHyperParameters = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = ParticleLeniaHyperParameters()

        self.locator = BlobLocator()
        # if config options set by decorator, override config
        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        self.uniform = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}",
            tensor_low=param_obj.tensor_low.numpy(),
            tensor_high=param_obj.tensor_high.numpy(),
            tensor_bound_low=param_obj.tensor_bound_low.numpy(),
            tensor_bound_high=param_obj.tensor_bound_high.numpy(),
        )

        self.uniform_mutator = partial(
            add_gaussian_noise,
            mean=ParticleLeniaDynamicalParameters(
                mu_k=0.5,
                sigma_k=0.1,
                w_k=0.01,
                mu_g=0.1,
                sigma_g=0.05,
                c_rep=0.1,
            ).to_tensor().numpy(),
            std=ParticleLeniaDynamicalParameters(
                mu_k=0.5,
                sigma_k=0.1,
                w_k=0.01,
                mu_g=0.1,
                sigma_g=0.05,
                c_rep=0.1,
            ).to_tensor().numpy(),
        )

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        """
        Takes input dictionary and overrides the
        premap_key with generated parameters for ParticleLenia.
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

    def sample(self) -> Dict:
        """
        Samples from the parameter map to yield the parameters ready
        for processing by the model (ParticleLenia).
        """
        # sample dynamical parameters
        p_dyn_tensor = self.uniform.sample()

        # convert to parameter objects
        dp = ParticleLeniaDynamicalParameters().from_tensor(torch.tensor(p_dyn_tensor))

        p_dict = {
            "dynamic_params": replace_torch_with_numpy(asdict(dp)),
        }

        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        """
        Takes the dictionary of only parameters outputted by
        the explorer and mutates them.
        """
        intermed_dict = deepcopy(parameter_dict)

        # mutate dynamic parameters
        dp = ParticleLeniaDynamicalParameters(**parameter_dict["dynamic_params"])
        dp_tensor = dp.to_numpy()
        mutated_dp_tensor = self.uniform_mutator(dp_tensor)

        intermed_dict["dynamic_params"] = asdict(
            ParticleLeniaDynamicalParameters().from_tensor(mutated_dp_tensor)
        )

        return intermed_dict
