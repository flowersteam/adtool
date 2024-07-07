import typing
from copy import deepcopy

import torch
from addict import Dict
from adtool.wrappers.BoxProjector import BoxProjector

from adtool.utils.misc.torch_utils import roll_n
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from examples.particlelenia.systems import ParticleLenia

EPS = 0.0001
DISTANCE_WEIGHT = 2  # 1=linear, 2=quadratic, ...

from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import cdist


import numpy as np

class ParticleLeniaStatistics(Leaf):
    """
    Outputs 17-dimensional embedding.
    """

    def __init__(
        self,
        system: ParticleLenia,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.locator = BlobLocator()

        self.premap_key = premap_key
        self.postmap_key = postmap_key




        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input_dict: typing.Dict) -> typing.Dict:
        """
        Compute statistics on Lenia's output
        Args:
            input: Lenia's output
            is_output_new_discovery: indicates if it is a new discovery
        Returns:
            Return a torch tensor in dict
        """

        intermed_dict = deepcopy(input_dict)


        # store raw output
        tensor = intermed_dict[self.premap_key].detach().clone()

        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(tensor)

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

    #  final_obs= torch.tensor(final_obs)





        # Max distance between particles
        max_distance = cdist(final_obs, final_obs).max()

        # Min distance between particles
        min_distance = cdist(final_obs, final_obs).min()

        # Average distance between particles
        avg_distance = cdist(final_obs, final_obs).mean()

        # Standard deviation of distance between particles
        std_distance = cdist(final_obs, final_obs).std()

        


        center_of_mass = final_obs.mean(axis=0)
        center_of_image = np.array([0, 0])
        distance_center_of_mass = np.linalg.norm(center_of_mass - center_of_image)

        # Compute the area of the convex hull of the particles
        hull = ConvexHull(final_obs)
        convex_area = hull.volume  # For 2D, `volume` is the area

        # Perimeter of the convex hull
        perimeter = hull.area






        return  np.array(
            (
                distance_center_of_mass,
                convex_area,
                perimeter,
                max_distance,
                min_distance,
                avg_distance,
                std_distance,
            
            
        ))
