import typing
from copy import deepcopy

import torch
from addict import Dict
from adtool.wrappers.BoxProjector import BoxProjector

from adtool.utils.misc.torch_utils import roll_n
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from examples.lenia.systems.Lenia import Lenia
from examples.stable_diffusion.systems.StableDiffusionPropagator import StableDiffusionPropagator

EPS = 0.0001


class SDStatistics(Leaf):
    """
    Outputs 17-dimensional embedding.
    """

    def __init__(
        self,
        system : StableDiffusionPropagator,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.locator = BlobLocator()

        self.premap_key = premap_key
        self.postmap_key = postmap_key


        self.SX = system.width
        self.SY = system.height

        # model
        self._statistic_names = [
            "mean",
            "contrast",
            "entropy",


        ]
        self._n_latents = len(self._statistic_names)

        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: typing.Dict) -> typing.Dict:
        """
        Compute statistics on System output
        Args:
            input: System output
        Returns:
            Return a torch tensor in dict
        """

        intermed_dict = deepcopy(input)

        # store raw output
        tensor = intermed_dict[self.premap_key].detach().clone()
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor
        del intermed_dict[self.premap_key]


        

        embedding = self._calc_static_statistics(tensor[0])

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        return self.projector.sample()

    def _calc_distance(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the distance between 2 embeddings in the latent space
        /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        return (embedding_a - embedding_b).pow(2).sum(-1).sqrt()

    def _calc_static_statistics(self, final_obs: torch.Tensor) -> torch.Tensor:
        """Calculates the final statistics for lenia last observation"""

        feature_vector = torch.zeros(self._n_latents)
        cur_idx = 0


        # 1. mean
        feature_vector[cur_idx] = final_obs.mean()
        cur_idx += 1

        # 2. contrast
        contrast = final_obs.max() - final_obs.min()
        feature_vector[cur_idx] = contrast
        cur_idx += 1

        # 3. entropy
        pre_entropy = final_obs
        pre_entropy = pre_entropy - pre_entropy.min()
        pre_entropy = pre_entropy / pre_entropy.max()
        pre_entropy = pre_entropy + EPS
        entropy = -torch.sum(pre_entropy * torch.log(pre_entropy))

        feature_vector[cur_idx] = entropy









        


        return feature_vector
