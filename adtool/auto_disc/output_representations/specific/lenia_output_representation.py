import typing

import torch
from addict import Dict
from adtool.auto_disc.output_representations import BaseOutputRepresentation

from adtool.auto_disc.utils.misc.torch_utils import roll_n
from adtool.auto_disc.utils.spaces import BoxSpace, DictSpace, DiscreteSpace
from adtool.auto_disc.utils.spaces.utils import ConfigParameterBinding, distance


# @StringConfigParameter(name="distance_function", possible_values=["L2"], default="L2")
# @IntegerConfigParameter(name="SX", default=256, min=1)
# @IntegerConfigParameter(name="SY", default=256, min=1)

from adtool.utils.expose_config.defaults import Defaults, defaults
from dataclasses import dataclass, field

@dataclass
class LeniaImageRepresentationConfig(Defaults):
    distance_function: str = defaults("L2", domain=["L2"])
    SX: int = defaults(256, min=1)
    SY: int = defaults(256, min=1)


@LeniaImageRepresentationConfig.expose_config()
class LeniaImageRepresentation(BaseOutputRepresentation):


    output_space = DictSpace(
        embedding=BoxSpace(
            low=0,
            high=10,
            shape=(ConfigParameterBinding("SX") * ConfigParameterBinding("SY"),),
        )
    )

    def __init__(self, wrapped_input_space_key: str = None, **kwargs) -> None:
        super().__init__("states", **kwargs)

    def map(
        self, input: typing.Dict, is_output_new_discovery: bool
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Flatten Lenia's output
        Args:
            input: Lenia's output
            is_output_new_discovery: indicates if it is a new discovery
        Returns:
            Return a torch tensor in dict
        """
        # filter low values
        filtered_im = torch.where(
            input.states[-1] > 0.2, input.states[-1], torch.zeros_like(input.states[-1])
        )

        # recenter
        mu_0 = filtered_im.sum()
        if mu_0.item() > 0:
            # implementation of meshgrid in torch
            x = torch.arange(self.config.SX)
            y = torch.arange(self.config.SY)
            xx = x.repeat(self.config.SY, 1)
            yy = y.view(-1, 1).repeat(1, self.config.SX)
            X = (xx - int(self.config.SX / 2)).float()
            Y = (yy - int(self.config.SY / 2)).float()

            centroid_x = ((X * filtered_im).sum() / mu_0).round().int().item()
            centroid_y = ((Y * filtered_im).sum() / mu_0).round().int().item()

            filtered_im = roll_n(filtered_im, 0, centroid_x)
            filtered_im = roll_n(filtered_im, 1, centroid_y)

        embedding = filtered_im.flatten()

        return {"embedding": embedding}

    def calc_distance(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute the distance between 2 embeddings in the latent space
        /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        if self.config.distance_function == "L2":
            # add regularizer to avoid dead outcomes
            dist = distance.calc_l2(embedding_a, embedding_b)

        else:
            raise NotImplementedError

        return dist
