
import dataclasses
import io
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sized

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
class FlowLeniaKernelGrowthDynamicalParameters:
    r: Union[int, float] = 0.5

    b: torch.Tensor = torch.tensor([0.1, 0.1, 0.1])
    w: torch.Tensor = torch.tensor([0.1, 0.1, 0.1])
    a: torch.Tensor = torch.tensor([0.1, 0.1, 0.1])
    h: Union[int, float] = 0.1


    m: Union[int, float] = 0.1
    s: Union[int, float] = 0.05

    def __post_init__(self):

        self.r = min(1.0, max(0.2, self.r))
        self.b = torch.clamp(self.b, min=0.001, max=1.0)
        self.w = torch.clamp(self.w, min=0.01, max=0.5)
        self.a = torch.clamp(self.a, min=0.0, max=1.0)
        self.m = min(0.5, max(0.05, self.m))
        self.s = min(0.18, max(0.001, self.s))
        self.h = min(1.0, max(0.01, self.h))

        if self.b.size() != (3,):
            raise ValueError("b must be a 3-vector.")
        if self.w.size() != (3,):
            raise ValueError("w must be a 3-vector.")
        if self.a.size() != (3,):
            raise ValueError("a must be a 3-vector.")



@dataclass
class FlowLeniaDynamicalParameters:

    R: Union[int, float] = 10
    KernelGrowths: List[FlowLeniaKernelGrowthDynamicalParameters ] = field(default_factory=lambda : [FlowLeniaKernelGrowthDynamicalParameters()]*40)


    def __init__(self, R: Union[int, float] = 10, KernelGrowths: List[FlowLeniaKernelGrowthDynamicalParameters] = [FlowLeniaKernelGrowthDynamicalParameters()]*40):
        self.R = min(25, max(2, R))

        #if KernelGrowths is not defined, set it to default value

        #check type of FlowLeniaKernelGrowthDynamicalParameters , and cast tho FlowLeniaKernelGrowthDynamicalParameters if it is not
        self.KernelGrowths = [KernelGrowth if isinstance(KernelGrowth, FlowLeniaKernelGrowthDynamicalParameters) else FlowLeniaKernelGrowthDynamicalParameters(**KernelGrowth) for KernelGrowth in KernelGrowths]

    # def __post_init__(self):
    #     self.R = min(25, max(2, self.R))

    #     print("self.KernelGrowths",self.KernelGrowths)

    #     #cast KernelGrowths to list
        




    def to_tensor(self) -> torch.Tensor:
        #converts FlowLeniaDynamicalParameters to tensor
        tensor = torch.tensor([self.R])
        for KernelGrowth in self.KernelGrowths:

            tensor = torch.cat((tensor, torch.tensor([KernelGrowth.r])), 0)
            tensor = torch.cat((tensor, KernelGrowth.b), 0)
            tensor = torch.cat((tensor, KernelGrowth.w), 0)
            tensor = torch.cat((tensor, KernelGrowth.a), 0)
            tensor = torch.cat((tensor, torch.tensor([KernelGrowth.h])), 0)
            tensor = torch.cat((tensor, torch.tensor([KernelGrowth.m])), 0)
            tensor = torch.cat((tensor, torch.tensor([KernelGrowth.s])), 0)

        return tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, ):
        #converts tensor to FlowLeniaDynamicalParameters (R and KernelGrowths)
        R= tensor[0]
        KernelGrowths = []
        i = 1
        while i < tensor.size()[0]:
            KernelGrowths.append(FlowLeniaKernelGrowthDynamicalParameters(r=tensor[i], b=tensor[i+1:i+4], w=tensor[i+4:i+7], a=tensor[i+7:i+10], h=tensor[i+10], m=tensor[i+11], s=tensor[i+12]))
            i += 13
        
        if i != tensor.size()[0]:
            raise ValueError("tensor size mismatch")

        return cls( R=R, KernelGrowths=KernelGrowths)


@dataclass
class FlowLeniaParameters:
    """Holds input parameters for Lenia model."""
    

    dynamic_params: FlowLeniaDynamicalParameters = field(default_factory=lambda : FlowLeniaDynamicalParameters())
    init_state: torch.Tensor = field(default_factory=lambda : torch.rand((10, 10,10)))


from pydantic import BaseModel
from pydantic.fields import Field
from adtool.utils.expose_config.expose_config import expose





@dataclass
class FlowLeniaHyperParameters:
    """Holds parameters to initialize Lenia model."""

    tensor_low: torch.Tensor = FlowLeniaDynamicalParameters().to_tensor()
    tensor_high: torch.Tensor = FlowLeniaDynamicalParameters().to_tensor()


    tensor_bound_low: torch.Tensor = FlowLeniaDynamicalParameters(

        R=2,
        KernelGrowths=[
            FlowLeniaKernelGrowthDynamicalParameters(
                r=0.2,
                b=torch.tensor([0.2, 0.2, 0.2]),
                w=torch.tensor([0.2, 0.2, 0.2]),
                a=torch.tensor([0.2, 0.2, 0.2]),
                h=0.2,
                m=0.2,
                s=0.01,
            )
        ]*40,


    ).to_tensor()
    tensor_bound_high: torch.Tensor = FlowLeniaDynamicalParameters(
            
        R=25,
        KernelGrowths=[
            FlowLeniaKernelGrowthDynamicalParameters(
                r=1.0,
                b=torch.tensor([1.0, 1.0, 1.0]),
                w=torch.tensor([0.5, 0.5, 0.5]),
                a=torch.tensor([1.0, 1.0, 1.0]),
                h=1.0,
                m=0.5,
                s=0.18,
            )
        ]*40,
    
        ).to_tensor()

    init_state_dim: Tuple[int, int,int] = (10, 10,10)
    cppn_n_passes: int = 2
