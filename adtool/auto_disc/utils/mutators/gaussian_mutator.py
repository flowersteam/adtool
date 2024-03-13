import numbers

import torch
from adtool.auto_disc.utils.mutators import BaseMutator


class GaussianMutator(BaseMutator):
    """
    Class to mutate a space with gaussian method
    """

    def __init__(self, mean: float, std: float) -> None:
        """
        Init what gaussian method need

        Args:
            mean: float
            std: float

        """
        self._mean = mean
        self._std = std

    def init_shape(self, shape: tuple = None) -> None:
        """
        Define the init shape

        Args:
            shape: tuple
        """
        super().init_shape(shape)
        if shape:
            if isinstance(self._mean, numbers.Number):
                self._mean = torch.full(shape, self._mean, dtype=torch.float64)
            if isinstance(self._std, numbers.Number):
                self._std = torch.full(shape, self._std, dtype=torch.float64)
        self.mean = torch.as_tensor(self._mean, dtype=torch.float64)
        self.std = torch.as_tensor(self._std, dtype=torch.float64)

    def __call__(self, x: torch.Tensor, mutate_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the gaussian mutation data

        Args:
            x: the data we want mutate
            mutate_mask: mask of mutation

        Returns:
            x: Data after being mutated
        """
        noise = torch.normal(self.mean, self.std)
        x = x.type(torch.float64) + mutate_mask * noise
        return x
