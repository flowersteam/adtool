from copy import deepcopy
from typing import Any, Dict, NamedTuple, Tuple

import torch
from adtool.maps.cppn import pytorchneat
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator


class NEATTuple(NamedTuple):
    # essentially an enum
    genome: str
    neat_config: str


class CPPNWrapper(Leaf):
    def __init__(
        self,
        premap_key: NEATTuple = NEATTuple("genome", "neat_config"),
        postmap_key: str = "init_state",
        postmap_shape: Tuple[int, int] = (10, 10),
        n_passes: int = 2,
    ):
        super().__init__()
        self.locator = BlobLocator()

        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.postmap_shape = postmap_shape

        self.n_passes = n_passes

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        # generate init_state and add to dict
        genome = intermed_dict[self.premap_key.genome]
        neat_config = intermed_dict[self.premap_key.neat_config]
        postmap_shape = self.postmap_shape
        n_passes = self.n_passes

        init_state = self._generate_init_state(
            genome, neat_config, postmap_shape, n_passes
        )

        intermed_dict[self.postmap_key] = init_state.detach()

        return intermed_dict

    @staticmethod
    def _generate_init_state(
        cppn_genome: Any, neat_config: Any, shape: Tuple[int, int], n_passes: int
    ) -> torch.Tensor:
        """
        Takes NEAT configuration and outputs the init_state tensor for Lenia
        """

        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(
            cppn_genome, neat_config
        )

        # configure output size
        cppn_output_height = int(shape[0])
        cppn_output_width = int(shape[1])

        cppn_input = pytorchneat.utils.create_image_cppn_input(
            (cppn_output_height, cppn_output_width),
            is_distance_to_center=True,
            is_bias=True,
        )
        cppn_output = initialization_cppn.activate(cppn_input, n_passes)
        cppn_net_output = (1.0 - cppn_output.abs()).squeeze()

        return cppn_net_output
