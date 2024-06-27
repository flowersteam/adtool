from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple

from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator

import numpy as np

class BoxProjector(Leaf):
    """
    Projects its input into a box space, i.e., the Cartesian product of N
    real-valued intervals, where the dimension N is set by the input dimension.

    Note that it's `map()` method essentially just passes the received input,
    but also adds a `sampler` key to the dict which an `Explorer` can use to
    sample from the space.
    """

    def __init__(
        self,
        premap_key: str,
        bound_upper: np.ndarray = np.array([float("inf")]),
        bound_lower:     np.ndarray = np.array([float("inf")]),
        init_low: Optional[np.ndarray] = None,
        init_high: Optional[np.ndarray] = None,
        tensor_shape: Optional[Tuple] = None,
    ) -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.bound_upper = bound_upper
        self.bound_lower = bound_lower

        # initialize data_shape if known
        if init_low is not None:
            self.tensor_shape = init_low.shape
        else:
            self.tensor_shape = tensor_shape

        self.low = init_low
        self.high = init_high

    def map(self, input: Dict) -> Dict:
        """
        Passes `input`, adding a `sampler` Callable item which `Explorer` can
        use to sample from the box space.
        """
        output = deepcopy(input)

        tensor_data = output[self.premap_key]

        # set tensor_shape dynamically
        if self.tensor_shape is None:
            self.tensor_shape = tensor_data.shape

        tensor_data = self._clamp_and_truncate(tensor_data)
        self._update_low_high(tensor_data)

        output[self.premap_key] = tensor_data

     #   print("tensor_data", tensor_data)   


        return output

    def sample(self) -> np.ndarray:
        dim = self.tensor_shape
        rand_nums = np.random.rand(*dim)

        dim_lengths = self.high - self.low
        sample = rand_nums * dim_lengths + self.low


        return sample

    def _clamp_and_truncate(self, data: np.ndarray) -> np.ndarray:
        # could use torch.clamp with updated pytorch version,
        # but in our version, torch.clamp doesn't support tensors
    #    clamped_data = torch.min(torch.max(data, self.bound_lower), self.bound_upper)
        clamped_data = np.minimum(np.maximum(data, self.bound_lower), self.bound_upper)
        # TODO: truncate dimensions

        return clamped_data


    def _update_low_high(self, data) -> None:
        """
        Update self.low and self.high which record the highest and lowest
        feature observations in the box space.
        """

        
        print(f"Converted data type: {type(data)}")  # Should print <class 'numpy.ndarray'>
        print(f"data shape: {data.shape}")
        
        if self.low is None:
            self.low = np.zeros_like(data)
        if self.high is None:
            self.high = np.zeros_like(data)

        print(f"self.low shape: {self.low.shape}")
        print(f"self.high shape: {self.high.shape}")

        # Convert types to ensure compatibility
        self.low = self.low.astype(np.float32)
        self.high = self.high.astype(np.float32)

        # Create masks for updates   """
        low_mask = np.less(data, self.low)
        high_mask = np.greater(data, self.high)

        print(f"low_mask shape: {low_mask.shape}")
        print(f"high_mask shape: {high_mask.shape}")

        # Check shapes before assignment to avoid shape mismatch error
        if data[low_mask].shape != self.low[low_mask].shape:
            print(f"data[low_mask].shape: {data[low_mask].shape}")
            print(f"self.low[low_mask].shape: {self.low[low_mask].shape}")
            raise ValueError("Shape mismatch in low mask assignment")
        
        if data[high_mask].shape != self.high[high_mask].shape:
            print(f"data[high_mask].shape: {data[high_mask].shape}")
            print(f"self.high[high_mask].shape: {self.high[high_mask].shape}")
            raise ValueError("Shape mismatch in high mask assignment")

        self.low[low_mask] = data[low_mask]
        self.high[high_mask] = data[high_mask]

        return
