from copy import deepcopy
from typing import Dict, Tuple

from adtool.maps.Map import Map
from adtool.wrappers.BoxProjector import BoxProjector
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator

import numpy as np

class MeanBehaviorMap(Map):
    """
    A simple `BehaviorMap` which merely extracts the mean.
    """

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "output",
        input_shape: Tuple[int] = (1,),
    ) -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.input_shape = input_shape  # unused by the module itself here
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict) -> Dict:
        # TODO: does not handle batches
        intermed_dict = deepcopy(input)

        # store raw output
        tensor = intermed_dict[self.premap_key].copy()
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor
        # remove original output item
        del intermed_dict[self.premap_key]

        # flatten to 1D
        tensor_flat = tensor.view(-1)

        # unsqueeze to ensure tensor rank is not 0
        #mean = torch.mean(tensor_flat, dim=0).unsqueeze(-1)
        mean=np.mean(tensor_flat.cpu().numpy())
        intermed_dict[self.postmap_key] = mean

        behavior_dict = self.projector.map(intermed_dict)

        return behavior_dict

    def sample(self) :
        return self.projector.sample()
