from typing import Any, Dict

import numpy as np
from adtool.utils.leaf.Leaf import Leaf


def add_gaussian_noise(
    input_tensor: np.ndarray,
    mean: np.ndarray = np.array([0.0]),
    std: np.ndarray = np.array([1.0]),
) -> np.ndarray:
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean, dtype=float)
    if not isinstance(std, np.ndarray):
        std = np.array(std, dtype=float)
    noise_unit = np.random.randn(input_tensor.size)
    noise_unit = noise_unit.reshape(input_tensor.shape)
    noise = noise_unit * std + mean
    return input_tensor + noise


def call_mutate_method(param_dict: Dict, param_map: Any = None) -> Dict:
    """
    If parameters are given as a complicated dict, then the parameter map which
    creates it must have a `mutate` method in order to mutate the underlying
    parameters controlling.
    """
    return param_map.mutate(param_dict)
