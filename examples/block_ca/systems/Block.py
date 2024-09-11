#!/usr/bin/env python3
import io
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import imageio
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from pydantic import BaseModel
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.nn import functional as F

from pydantic.fields import Field
from adtool.systems.System import System
import numpy as np

from typing import Any

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class BlockParams:
    a: float  # Parameter a
    b: float  # Parameter b
    c: float  # Parameter c
    d: float  # Parameter d
    p: float  # Parameter p
    q: float  # Parameter q
    r: float  # Parameter r
    s: float  # Parameter s


class BlockSimulation:
    def __init__(
        self,
        height=512,
        width=512,
        num_inference_steps: int = 1000,
        initial_condition_seed=0,
        device: str = TORCH_DEVICE
    ) -> None:
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.initial_condition_seed = initial_condition_seed
        self.device = device


    def update_block(self, block):
        # Flatten the block
        block = block.flatten()

        # Apply the transformation using the matrix
        new_block = torch.matmul(self.matrix, block)

        # Apply non-linear activation (tanh)
     #   new_block = torch.tanh(new_block)

        # Reshape the block back to 2x2
        new_block = new_block.view(2, 2)

        return new_block


    def _initialize_conditions(self):
        # Set the seed for reproducibility

        self.frames = []

        # 4D rotation matrices (converted to torch tensors)
        matrix1 = torch.tensor([[self.params.a, -self.params.b, -self.params.c, -self.params.d],
                                [self.params.b, self.params.a, -self.params.d, self.params.c],
                                [self.params.c, self.params.d, self.params.a, -self.params.b],
                                [self.params.d, -self.params.c, self.params.b, self.params.a]], device=self.device)

        matrix2 = torch.tensor([[self.params.p, -self.params.q, -self.params.r, -self.params.s],
                                [self.params.q, self.params.p, self.params.s, -self.params.r],
                                [self.params.r, -self.params.s, self.params.p, self.params.q],
                                [self.params.s, self.params.r, -self.params.q, self.params.p]], device=self.device)

        self.matrix = torch.matmul(matrix1, matrix2)

        # Initialize the grid state
        self.state = torch.zeros((self.width, self.height), device=self.device)
        # random noise in the center of one fifth of the grid
        # self.state[self.width // 2 - self.width // 10:self.width // 2 + self.width // 10,
        #              self.height // 2 - self.height // 10:self.height // 2 + self.height // 10] = torch.rand(
        #     (self.width // 5, self.height // 5), device=self.device
        # )

        # The expanded size of the tensor (24) must match the existing size (25) at non-singleton dimension 1.  Target sizes: [24, 24].  Tensor sizes: [25, 25]
        
        k=15
        # self.state[self.width // 2 - self.width // (2*k):self.width // 2 + self.width // (2*k)
                   
        #            + (self.width % 2) 
        #            ,
        #              self.height // 2 - self.height // (2*k):self.height // 2 + self.height // (2*k)

        #              + (self.height % 2)
                     
        #              ] = torch.rand(
        #     (self.width // k, self.height // k), device=self.device
        # )

        # just 4 random points in the middle of the grid
        self.state[self.width // 2 - 1, self.height // 2 - 1] = torch.rand(1, device=self.device)
        self.state[self.width // 2 - 1, self.height // 2] = torch.rand(1, device=self.device)
        self.state[self.width // 2, self.height // 2 - 1] = torch.rand(1, device=self.device)
        self.state[self.width // 2, self.height // 2] = torch.rand(1, device=self.device)







    def _update(self, grid, step):
        offset = (step % 2)
        new_grid = grid.clone()

        for i in range(offset, self.width - 1, 2):
            for j in range(offset, self.height - 1, 2):
                block = grid[i:i+2, j:j+2]
                new_grid[i:i+2, j:j+2] = self.update_block(block)

        return new_grid


    def update(self):
        """Integrate the resulting system of equations using the new matrix parameters."""
        for i in range(self.num_inference_steps):
            self.state = self._update(self.state, i)
            self.frames.append(self.state.cpu().numpy())



    def _decode_image(self):
        # Just use self.state as the grayscale image
        return self.state.cpu()#.numpy()  # Convert back to numpy for image processing

    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        self.params = BlockParams(**input["params"]["dynamic_params"])

      #  if fix_seed:
      #      torch.manual_seed(self.initial_condition_seed)
        self._initialize_conditions()
        self.update()
        input["output"] = self._decode_image()
        return input

    def render(self, data_dict, mode: str = "PIL_image") -> Tuple[bytes,str]:
        # ignores data_dict, as the render is based on self.orbit
        # in which only the last state is stored in data_dict["output"]

        colormap = create_colormap(
            np.array(
                [
                    [255, 255, 255],
                    [119, 255, 255],
                    [23, 223, 252],
                    [0, 190, 250],
                    [0, 158, 249],
                    [0, 142, 249],
                    [81, 125, 248],
                    [150, 109, 248],
                    [192, 77, 247],
                    [232, 47, 247],
                    [255, 9, 247],
                    [200, 0, 84],
                ]
            )
            / 255
            * 8
        )
        im_array = []
        for img in self.frames:
            # need to squeeze leading dimensions
            parsed_img = img.squeeze()
            im = im_from_array_with_colormap(parsed_img, colormap)
            im_array.append(im.convert("RGB"))

        if mode == "human":
            matplotlib.use("TkAgg")
            fig = plt.figure(figsize=(4, 4))
            animation = FuncAnimation(
                fig, lambda frame: plt.imshow(frame), frames=im_array
            )
            plt.axis("off")
            plt.tight_layout()
            return plt.show()
        elif mode == "PIL_image":
            byte_img = io.BytesIO()
            imageio.mimwrite(
                byte_img, im_array, "mp4", fps=20, output_params=["-f", "mp4"]
            )
            return [(byte_img.getvalue(), "mp4")]
        else:
            raise NotImplementedError



class GenerationParams(BaseModel):
    height: int = Field(512, ge=64, le=1024)
    width: int = Field(512, ge=64, le=1024)
    num_inference_steps: int = Field(1000, ge=1, le=10000)
    initial_condition_seed: int = Field(0, ge=0, le=999999)


from adtool.utils.expose_config.expose_config import expose

@expose
class Block(BlockSimulation):
    config = GenerationParams

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


def create_colormap(colors: np.ndarray, is_marker_w: bool = True) -> List[int]:

    MARKER_COLORS_W = [0x5F, 0x5F, 0x5F, 0x7F, 0x7F, 0x7F, 0xFF, 0xFF, 0xFF]
    MARKER_COLORS_B = [0x9F, 0x9F, 0x9F, 0x7F, 0x7F, 0x7F, 0x0F, 0x0F, 0x0F]
    nval = 253
    ncol = colors.shape[0]
    colors = np.vstack((colors, np.array([[0, 0, 0]])))
    v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 252 252 252]
    i = np.array(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
    k = v / (nval - 1) * (ncol - 1)  # interpolate between 0 .. ncol-1
    k1 = k.astype(int)
    c1, c2 = colors[k1, i], colors[k1 + 1, i]
    c = (k - k1) * (c2 - c1) + c1  # interpolate between c1 .. c2
    return np.rint(c / 8 * 255).astype(int).tolist() + (
        MARKER_COLORS_W if is_marker_w else MARKER_COLORS_B
    )


def im_from_array_with_colormap(np_array: np.ndarray, colormap: List[int]) -> Image:
    """
    Function that transforms the color palette of a PIL image

    input:
        - image: the PIL image to transform
        - colormap: the desired colormap
    output: the transformed PIL image
    """
    image_array = np.uint8(np_array.astype(float) * 252.0)
    transformed_image = Image.fromarray(image_array)
    transformed_image.putpalette(colormap)

    return transformed_image
