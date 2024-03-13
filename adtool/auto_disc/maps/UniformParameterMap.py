from copy import deepcopy
from typing import Dict, List, Tuple, Union

import torch
from adtool.auto_disc.maps.Map import Map
from adtool.auto_disc.wrappers.BoxProjector import BoxProjector
from adtool.auto_disc.wrappers.SaveWrapper import SaveWrapper
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator


class UniformParameterMap(Map):
    """
    A simple `ParameterMap` which generates parameters according to a uniform
    distribution over a box.
    """

    def __init__(
        self,
        premap_key: str = "params",
        tensor_low: Union[torch.Tensor, List[float]] = torch.tensor([0.0]),
        tensor_high: Union[torch.Tensor, List[float]] = torch.tensor([0.0]),
        tensor_bound_low: Union[torch.Tensor, List[float]] = torch.tensor(
            [float("-inf")]
        ),
        tensor_bound_high: Union[torch.Tensor, List[float]] = torch.tensor(
            [float("inf")]
        ),
        override_existing: bool = True,
    ):
        # TODO: put indication that tensor_low and high must be set
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key

        # convert all Union types to tensors
        if not isinstance(tensor_low, torch.Tensor):
            tensor_low = torch.tensor(tensor_low)
        if not isinstance(tensor_high, torch.Tensor):
            tensor_high = torch.tensor(tensor_high)
        if not isinstance(tensor_bound_low, torch.Tensor):
            tensor_bound_low = torch.tensor(tensor_bound_low)
        if not isinstance(tensor_bound_high, torch.Tensor):
            tensor_bound_high = torch.tensor(tensor_bound_high)

        # ensure no tensor is of size 0 by unsqueezing
        if tensor_low.size() == torch.Size([]):
            tensor_low = tensor_low.unsqueeze(0)
        if tensor_high.size() == torch.Size([]):
            tensor_high = tensor_high.unsqueeze(0)
        if tensor_bound_low.size() == torch.Size([]):
            tensor_bound_low = tensor_bound_low.unsqueeze(0)
        if tensor_bound_high.size() == torch.Size([]):
            tensor_bound_high = tensor_bound_high.unsqueeze(0)

        if tensor_low.size() != tensor_high.size():
            raise ValueError("tensor_low and tensor_high must be same shape.")
        if tensor_bound_low.size() != tensor_bound_high.size():
            raise ValueError(
                "tensor_bound_low and tensor_bound_high must be same shape."
            )
        self.postmap_shape = tensor_low.size()
        # self.history_saver = SaveWrapper()

        self.projector = BoxProjector(
            premap_key=premap_key,
            init_high=tensor_high,
            init_low=tensor_low,
            bound_lower=tensor_bound_low,
            bound_upper=tensor_bound_high,
        )

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        """
        map() takes an input dict of metadata and adds the
        `params` key with a sample if it does not exist
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

        param_dict = self.projector.map(intermed_dict)
        # params_dict = self.history_saver.map(intermed_dict)

        return param_dict

    def sample(self) -> torch.Tensor:
        data_shape = self.postmap_shape
        dimensions_to_keep = data_shape[0]
        sample = self.projector.sample()
        return sample[:dimensions_to_keep]

    # def get_tensor_history(self) -> torch.Tensor:
    #     tensor_history = \
    #         self.history_saver.buffer[0][self.premap_key].unsqueeze(0)
    #     for dict in self.history_saver.buffer[1:]:
    #         tensor_history = torch.cat(
    #             (tensor_history, dict[self.premap_key].unsqueeze(0)), dim=0)
    #     return tensor_history
