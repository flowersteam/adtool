
import dataclasses
import io
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from adtool.utils.misc.torch_utils import (
    SphericPad,
    complex_mult_torch,
    roll_n,
    soft_clip,
)
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from matplotlib.animation import FuncAnimation
from numpy import ndarray
from PIL import Image


@dataclass
class LeniaDynamicalParameters:
    R: Union[int, float] = 0
    T: float = 1.0
    m: float = 0.0
    s: float = 0.001
    b: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0])

    def __post_init__(self):
        # convert out of tensors
        if isinstance(self.R, torch.Tensor):
            self.R = self.R.item()
        if isinstance(self.T, torch.Tensor):
            self.T = self.T.item()
        if isinstance(self.m, torch.Tensor):
            self.m = self.m.item()
        if isinstance(self.s, torch.Tensor):
            self.s = self.s.item()

        if isinstance(self.b, np.ndarray):
            self.b = torch.from_numpy(self.b)


        # check constraints
        if isinstance(self.R, float):
            self.R = min(19,round(self.R))

        self.m = min(1.0, max(0.0, self.m))

        self.s = min(0.3, max(0.001, self.s))
        
        self.T =min(10.0, max(1.0, self.T))

        if  (4,) != self.b.shape:
            raise ValueError("b must be a 4-vector.")
        self.b = torch.clamp(self.b, min=0.0, max=1.0)




    def to_tensor(self) -> torch.Tensor:
        return torch.cat(
            (torch.tensor([self.R, self.T]), torch.tensor([self.m, self.s]), self.b)
        )
    
    def to_numpy(self) -> np.ndarray:
        return np.concatenate(
            (np.array([self.R, self.T]), np.array([self.m, self.s]), self.b.numpy())
        )
    
    @classmethod
    def from_numpy(cls, np_array: np.ndarray):
        r = np_array[0]
        t = np_array[1]
        m = np_array[2]
        s = np_array[3]
        b = np_array[4:8]
        return cls(R=r, T=t, m=m, s=s, b=b)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        r = tensor[0].item()
        t = tensor[1].item()
        m = tensor[2].item()
        s = tensor[3].item()
        b = tensor[4:8]
        return cls(R=r, T=t, m=m, s=s, b=b)


@dataclass
class LeniaParameters:
    """Holds input parameters for Lenia model."""

    dynamic_params: LeniaDynamicalParameters = field(default_factory=lambda : LeniaDynamicalParameters())
    init_state: torch.Tensor = field(default_factory=lambda : torch.rand((10, 10)))


from pydantic import BaseModel
from pydantic.fields import Field
from adtool.utils.expose_config.expose_config import expose




@dataclass
class LeniaHyperParameters:
    """Holds parameters to initialize Lenia model."""

    tensor_low: torch.Tensor = LeniaDynamicalParameters().to_tensor()
    tensor_high: torch.Tensor = LeniaDynamicalParameters().to_tensor()
    tensor_bound_low: torch.Tensor = LeniaDynamicalParameters(

        R=0.0,
        T=1.0,
        b=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        m=0.0,
        s=0.001




    ).to_tensor()
    tensor_bound_high: torch.Tensor = LeniaDynamicalParameters(
            
            R=20,
            T=20,
            b=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            m=1.0,
            s=0.3
    
        ).to_tensor()

    init_state_dim: Tuple[int, int] = (10, 10)
    cppn_n_passes: int = 2
