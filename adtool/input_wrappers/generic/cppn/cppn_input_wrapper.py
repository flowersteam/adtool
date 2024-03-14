import typing
from typing import Any, Dict

import torch
from adtool.input_wrappers import BaseInputWrapper
from adtool.input_wrappers.generic.cppn import pytorchneat
from adtool.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from adtool.utils.spaces import DictSpace

from adtool.utils.expose_config.defaults import Defaults, defaults
from dataclasses import dataclass, field

@dataclass
class CppnInputWrapperConfig(Defaults):
    n_passes: int = defaults(2, min=1)


#@IntegerConfigParameter(name="n_passes", default=2, min=1)
@CppnInputWrapperConfig.expose_config()
class CppnInputWrapper(BaseInputWrapper):
    """Base class to map the parameters sent by the explorer to the system's input space"""

    

    input_space = DictSpace(genome=CPPNGenomeSpace())

    def __init__(self, wrapped_output_space_key: str, **kwargs) -> None:
        super().__init__(wrapped_output_space_key, **kwargs)

    def map(
        self, input: Dict[str, Any], is_input_new_discovery: bool, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Map the input parameters (from the explorer) to the cppn output parameters (sytem input)

        Args:
            parameters: cppn input parameters
            is_input_new_discovery: indicates if it is a new discovery
        Returns:
            parameters: parameters after map to match system input
        """
        cppn_genome = input["genome"]
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(
            cppn_genome, self.input_space["genome"].neat_config
        )

        cppn_output_height = int(
            self.output_space[self._wrapped_output_space_key].shape[1]
        )
        cppn_output_width = int(
            self.output_space[self._wrapped_output_space_key].shape[0]
        )

        cppn_input = pytorchneat.utils.create_image_cppn_input(
            (cppn_output_height, cppn_output_width),
            is_distance_to_center=True,
            is_bias=True,
        )
        cppn_output = initialization_cppn.activate(cppn_input, self.config.n_passes)
        cppn_net_output = (1.0 - cppn_output.abs()).squeeze()

        input[self._wrapped_output_space_key] = cppn_net_output
        del input["genome"]
        return input
