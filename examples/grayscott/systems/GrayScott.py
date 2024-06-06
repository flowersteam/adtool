#!/usr/bin/env python3
import io
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple

import imageio
import numpy as np
from pydantic import BaseModel
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.nn import functional as F

from pydantic.fields import Field
from adtool.systems.System import System

from typing import Any

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import numpy as np
from dataclasses import dataclass
from PIL import Image
import io
from typing import Dict, Any, Tuple
from tqdm import tqdm

@dataclass
class GrayScottParams:
    F: float  # Feed rate
    k: float  # Kill rate
    # Du: float  # Diffusion rate of U
    # Dv: float  # Diffusion rate of V

class GrayScottSimulation:
    def __init__(
        self,
        height=512,
        width=512,
        num_inference_steps: int = 1000,
            initial_condition_seed=0,
        device: str = 'cpu'
    ) -> None:
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps

        self.initial_condition_seed = initial_condition_seed
        self.device = device



    def _initialize_conditions(self):


        np.random.seed(self.initial_condition_seed)

        self.u = torch.ones((self.height, self.width), dtype=torch.float32, device=self.device)
        self.v = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)

        self.frames = []

        perturbation_size = self.height // 10

        self.u += 0.02*np.random.random((self.height, self.width))
        self.v += 0.02*np.random.random((self.height, self.width))

        # Initial perturbation in the center
        center = (self.height // 2, self.width // 2)
        self.u[center[0]-perturbation_size:center[0]+perturbation_size, center[1]-perturbation_size:center[1]+perturbation_size] = 0.50
        self.v[center[0]-perturbation_size:center[0]+perturbation_size, center[1]-perturbation_size:center[1]+perturbation_size] = 0.25


    def _laplacian(self, x):
        lap = -4 * x
        lap += torch.roll(x, shifts=1, dims=0)
        lap += torch.roll(x, shifts=-1, dims=0)
        lap += torch.roll(x, shifts=1, dims=1)
        lap += torch.roll(x, shifts=-1, dims=1)
        return lap

    def update(self):
        """Integrate the resulting system of equations using the Euler method"""
        for _ in range(self.num_inference_steps):
            uvv = self.u * self.v * self.v
            u_lap = self._laplacian(self.u)
            v_lap = self._laplacian(self.v)

            self.u += self.params.Du * u_lap - uvv + self.params.F * (1 - self.u)
            self.v += self.params.Dv * v_lap + uvv - (self.params.F + self.params.k) * self.v

            # Capture frame
            self.frames.append(self._decode_image().cpu().numpy().astype(np.uint8))
            # if the last two frames are the same, break
            if len(self.frames) > 1 and np.all(self.frames[-1] == self.frames[-2]):
                break


    def _decode_image(self):
        # just use self.u as the grayscale image
        image = torch.stack([self.u, self.u, torch.zeros_like(self.u)], dim=2)
        image = (image / image.max() * 255).clamp(0, 255)
        return image

    def map(self, input: Dict, fix_seed: bool = True) -> Dict:

        #         params: GrayScottParams = GrayScottParams(F=0.04, k=0.06, Du=0.16, Dv=0.08),

        # input {'params': {'dynamic_params': {'F': 0.8822692632675171, 'k': 0.9150039553642273, 'Du': 0.38286375999450684, 'Dv': 0.9593056440353394}}, 'equil': 1}
        self.params = GrayScottParams(**input["params"]["dynamic_params"])

        # self.params.F = 0.035
        # self.params.k = 0.065
        self.params.Du = 0.14
        self.params.Dv = 0.06


        # self.params.Du = 0.16
        # self.params.Dv = 0.08
        # self.params.F = 0.060
        # self.params.k = 0.062

        # self.params.F=0.039
        # self.params.k=0.058
        print(self.params.F, self.params.k)

        if fix_seed:
            np.random.seed(self.initial_condition_seed)
        self._initialize_conditions()
        self.update()
        input["output"] = self._decode_image()
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        im_array = [frame for frame in self.frames]
        
        byte_img = io.BytesIO()
        imageio.mimwrite(byte_img, im_array, format='mp4', fps=self.num_inference_steps/10)
        byte_img.seek(0)
        return byte_img.getvalue(), "mp4"

class GenerationParams(BaseModel):
    height: int = Field(512, ge=64, le=1024)
    width: int = Field(512, ge=64, le=1024)
    num_inference_steps: int = Field(1000, ge=1, le=10000)
    initial_condition_seed: int = Field(0, ge=0, le=999999)

from adtool.utils.expose_config.expose_config import expose

@expose
class GrayScott(GrayScottSimulation):
    config = GenerationParams

    def __init__(
        self,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
