from copy import deepcopy
from typing import Dict, Tuple

import torch
from adtool.auto_disc.maps.Map import Map
from adtool.auto_disc.wrappers.BoxProjector import BoxProjector
from adtool.auto_disc.wrappers.SaveWrapper import SaveWrapper
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator


class MeanBehaviorMap(Map):
    """
    A simple `BehaviorMap` which merely extracts the mean.
    """

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "output",
        input_shape: Tuple[int] = (1),
    ) -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.input_shape = input_shape  # unused by the module itself here
        # self.history_saver = SaveWrapper()
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict) -> Dict:
        # TODO: does not handle batches
        intermed_dict = deepcopy(input)

        # store raw output
        tensor = intermed_dict[self.premap_key].detach().clone()
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor
        # remove original output item
        del intermed_dict[self.premap_key]

        # flatten to 1D
        tensor_flat = tensor.view(-1)

        # unsqueeze to ensure tensor rank is not 0
        mean = torch.mean(tensor_flat, dim=0).unsqueeze(-1)
        intermed_dict[self.postmap_key] = mean

        behavior_dict = self.projector.map(intermed_dict)
        # behavior_dict = self.history_saver.map(projected_dict)

        return behavior_dict

    def sample(self) -> torch.Tensor:
        return self.projector.sample()

    # def get_tensor_history(self):
    #     tensor_history = \
    #         self.history_saver.buffer[0][self.premap_key].unsqueeze(0)
    #     for dict in self.history_saver.buffer[1:]:
    #         tensor_history = torch.cat(
    #             (tensor_history, dict[self.premap_key].unsqueeze(0)),
    #             dim=0)
    #     return tensor_history
