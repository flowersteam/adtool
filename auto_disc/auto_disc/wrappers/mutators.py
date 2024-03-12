from typing import Any, Dict

import torch
from auto_disc.utils.leaf.Leaf import Leaf


def add_gaussian_noise(
    input_tensor: torch.Tensor,
    mean: torch.Tensor = torch.tensor([0.0]),
    std: torch.Tensor = torch.tensor([1.0]),
) -> torch.Tensor:
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=float)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=float)
    noise_unit = torch.randn(input_tensor.size())
    noise = noise_unit * std + mean
    return input_tensor + noise


def call_mutate_method(param_dict: Dict, param_map: Any = None) -> Dict:
    """
    If parameters are given as a complicated dict, then the parameter map which
    creates it must have a `mutate` method in order to mutate the underlying
    parameters controlling.
    """
    return param_map.mutate(param_dict)
