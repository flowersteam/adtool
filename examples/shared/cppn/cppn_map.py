"""Map CPPN genomes to initialization tensors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, NamedTuple, Tuple

import torch

from adtool.maps.base import BaseMap, Payload
from examples.shared.cppn import pytorchneat


class NEATKeys(NamedTuple):
    genome: str = "genome"
    neat_config: str = "neat_config"


class CPPNMap(BaseMap):
    """Generate an initial state from a NEAT genome stored in a payload."""

    def __init__(
        self,
        premap_key: NEATKeys = NEATKeys(),
        postmap_key: str = "init_state",
        postmap_shape: Tuple[int, ...] = (10, 10),
        n_passes: int = 2,
    ) -> None:
        super().__init__(premap_key=str(premap_key), postmap_key=postmap_key)
        self.neat_keys = premap_key
        self.postmap_shape = postmap_shape
        self.n_passes = n_passes

    def map(self, payload: Payload, override_existing: bool = True) -> Payload:
        del override_existing
        output = deepcopy(payload)
        output[self.postmap_key] = self._generate_init_state(
            output[self.neat_keys.genome],
            output[self.neat_keys.neat_config],
            self.postmap_shape,
            self.n_passes,
        ).detach()
        return output

    @staticmethod
    def _generate_init_state(
        cppn_genome: Any, neat_config: Any, shape: Tuple[int, ...], n_passes: int
    ) -> torch.Tensor:
        network = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, neat_config)
        cppn_input = pytorchneat.utils.create_image_cppn_input(
            shape, is_distance_to_center=True, is_bias=True
        )
        return 1.0 - network.activate(cppn_input, n_passes).abs()
