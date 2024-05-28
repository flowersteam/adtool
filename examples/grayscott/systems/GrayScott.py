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


TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



@dataclass
class GrayScottParams:
    F: float  # Feed rate
    k: float  # Kill rate
    Du: float  # Diffusion rate of U
    Dv: float  # Diffusion rate of V


class GrayScottSimulation(System):
    def __init__(
        self,
        height=512,
        width=512,
        num_inference_steps: int = 1000,
        params: GrayScottParams = GrayScottParams(F=0.04, k=0.06, Du=0.16, Dv=0.08),
        initial_condition_seed=0,
        *args, **kwargs
    ) -> None:
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.params = params
        self.initial_condition_seed = initial_condition_seed

        self.u = np.ones((self.height, self.width), dtype=np.float32)
        self.v = np.zeros((self.height, self.width), dtype=np.float32)

        self._initialize_conditions()

    def _initialize_conditions(self):
        np.random.seed(self.initial_condition_seed)
        perturbation_size = self.height // 10

        # Initial perturbation in the center
        center = (self.height // 2, self.width // 2)
        self.u[center[0]-perturbation_size:center[0]+perturbation_size, center[1]-perturbation_size:center[1]+perturbation_size] = 0.50
        self.v[center[0]-perturbation_size:center[0]+perturbation_size, center[1]-perturbation_size:center[1]+perturbation_size] = 0.25

    def update(self):
        laplacian = lambda z: (
            -z +
            0.2 * (np.roll(z, 1, 0) + np.roll(z, -1, 0) +
                   np.roll(z, 1, 1) + np.roll(z, -1, 1)) +
            0.05 * (np.roll(np.roll(z, 1, 0), 1, 1) + np.roll(np.roll(z, 1, 0), -1, 1) +
                    np.roll(np.roll(z, -1, 0), 1, 1) + np.roll(np.roll(z, -1, 0), -1, 1))
        )

        for _ in tqdm(range(self.num_inference_steps)):
            Lu = laplacian(self.u)
            Lv = laplacian(self.v)
            uvv = self.u * self.v * self.v
            self.u += self.params.Du * Lu - uvv + self.params.F * (1 - self.u)
            self.v += self.params.Dv * Lv + uvv - (self.params.F + self.params.k) * self.v

    def _decode_image(self):
        image = np.stack([self.u, self.v, np.zeros_like(self.u)], axis=2)
        image = (image / image.max() * 255).astype(np.uint8)
        return Image.fromarray(image)
    
    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        if fix_seed:
            np.random.seed(self.initial_condition_seed)
        self._initialize_conditions()
        self.update()
        input["output"] = self._decode_image()
        return input

    def render(self) -> Tuple[bytes, str]:
        self.update()
        img = self._decode_image()
        byte_img = io.BytesIO()
        img.save(byte_img, format='PNG')
        return byte_img.getvalue(), "png"


class GenerationParams(BaseModel):
    height: int = Field(512, ge=64, le=1024)
    width: int = Field(512, ge=64, le=1024)
    num_inference_steps: int = Field(1000, ge=1, le=10000)
    # F: float = Field(0.04, ge=0.0, le=0.1)
    # k: float = Field(0.06, ge=0.0, le=0.1)
    # Du: float = Field(0.16, ge=0.0, le=1.0)
    # Dv: float = Field(0.08, ge=0.0, le=1.0)
    initial_condition_seed: int = Field(0, ge=0, le=999999)


class GrayScott(GrayScottSimulation):
    config = GenerationParams

    def __init__(
        self,
        *args, **kwargs
    ) -> None:
        
        print("GrayScott.__init__")
        print("args:", args)
        print("kwargs:", kwargs)

        super().__init__(*args, **kwargs)

