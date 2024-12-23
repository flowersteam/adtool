import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from io import StringIO
from typing import Dict, Optional, Tuple

import torch
from adtool.systems import System
from examples.flowlenia.systems.FlowLenia import FlowLenia
from examples.flowlenia.systems.FlowLeniaParameters import FlowLeniaDynamicalParameters, FlowLeniaHyperParameters, FlowLeniaKernelGrowthDynamicalParameters
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.wrappers.mutators import add_gaussian_noise
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
import sys
from adtool.utils.misc.torch_utils import replace_torch_with_numpy









class FlowLeniaParameterMap(Leaf):
    """
    Due to the complexities of initializing Lenia parameters,
    it's easier to make this custom parameter map.
    """

    def __init__(
        self,
        system: FlowLenia,
        premap_key: str = "params",
        param_obj: FlowLeniaHyperParameters = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = FlowLeniaHyperParameters.from_nb_k(system.nb_k)

        self.locator = BlobLocator()
        # if config options set by decorator, override config
        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        self.uniform = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}",
            tensor_low=param_obj.tensor_low.numpy() ,
            tensor_high=param_obj.tensor_high.numpy(),
            tensor_bound_low=param_obj.tensor_bound_low.numpy(),
            tensor_bound_high=param_obj.tensor_bound_high.numpy(),
        )


        self.uniform_mutator = partial(
            add_gaussian_noise,
            mean=FlowLeniaDynamicalParameters(
                R = 0.2,
                KernelGrowths = [
                    FlowLeniaKernelGrowthDynamicalParameters(
                        r = 0.2,
                        b = torch.tensor([0.2, 0.2, 0.2]),
                        w = torch.tensor([0.2, 0.2, 0.2]),
                        a = torch.tensor([0.2, 0.2, 0.2]),
                        h = 0.2,
                        m = 0.2,
                        s = 0.01
                    )
                ] * system.nb_k
            ).to_tensor().numpy(),
            
            

    

            std=FlowLeniaDynamicalParameters(

                R = 0.2,
                KernelGrowths = [
                    FlowLeniaKernelGrowthDynamicalParameters(
                        r = 0.2,
                        b = torch.tensor([0.2, 0.2, 0.2]),
                        w = torch.tensor([0.2, 0.2, 0.2]),
                        a = torch.tensor([0.2, 0.2, 0.2]),
                        h = 0.2,
                        m = 0.2,
                        s = 0.01
                    )
                ] * system.nb_k
    ).to_tensor().numpy(),
        )

        self.SX = param_obj.init_state_dim[1]
        self.SY = param_obj.init_state_dim[0]
        self.cppn_n_passes = param_obj.cppn_n_passes

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

    def sample(self) -> Dict:
        """
        Samples from the parameter map to yield the parameters ready
        for processing by the model (LeniaCPPN), i.e., including the genome.
        """
        # sample dynamical parameters
        p_dyn_tensor = self.uniform.sample()



        # convert to parameter objects
        dp = FlowLeniaDynamicalParameters().from_numpy(p_dyn_tensor)


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
        dp = FlowLeniaDynamicalParameters(**parameter_dict["dynamic_params"])
        dp_tensor = dp.to_numpy()
        mutated_dp_tensor = self.uniform_mutator(dp_tensor)

        intermed_dict["dynamic_params"] = asdict(
            FlowLeniaDynamicalParameters().from_numpy(mutated_dp_tensor)
        )



        return intermed_dict
