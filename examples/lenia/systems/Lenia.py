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

from examples.lenia.systems.LeniaParameters import LeniaDynamicalParameters, LeniaParameters





def patched_rfft(input, signal_ndim: int = 1, onesided: bool = True):
    dim = (-3,-2,-1)[-signal_ndim:]

    def partial_application(x, f):
        """Applies f and returns complex tensor encoded as a real-valued tensor
        with an extra dimension of size 2."""
        return torch.view_as_real(f(x, dim=dim))

    if onesided:
        return partial_application(input, torch.fft.rfftn)
    else:
        # non-onesided FFT is equivalent to the normal one
        return partial_application(input, torch.fft.fftn)


def patched_irfft(input, signal_ndim: int = 1, onesided: bool = True):
    dim = (-3,-2,-1)[-signal_ndim:]

    def partial_application(x, f):
        """Applies f and returns complex tensor encoded as a real-valued tensor
        with an extra dimension of size 2."""
        # take only the real part, as this is what we expect by the inverse
        # FFT to a real-valued signal (numerically, it will only be real within
        # machine epsilon, so we explicitly cast)
        return torch.real(f(x, dim=dim))

    # because the torch.fft functions expect a complex Tensor
    input = torch.view_as_complex(input)

    if onesided:
        return partial_application(input, torch.fft.irfftn)
    else:
        # non-onesided FFT is equivalent to the normal one
        return partial_application(input, torch.fft.ifftn)


torch.rfft = patched_rfft
torch.irfft = patched_irfft



from adtool.systems.System import System

class Lenia(System):


    def __init__(self, SX=256,
                 SY=256,
                 dt=.2,
                 dd=5,
                 sigma=.65,
                 mix="softmax",
                 final_step=200,
                 scale_init_state=1, **kwargs):
        super().__init__()
        self.SX= SX
        self.SY= SY
        self.dt= dt
        self.dd= dd
        self.sigma= sigma
        self.mix= mix
        self.scale_init_state= scale_init_state
        self.final_step= final_step
        self.locator = BlobLocator()
        self.orbit = torch.empty(
                (self.final_step, 1, 1, self.SX, self.SY),
            requires_grad=False,
        )

    def map(self, input: Dict) -> Dict:
        params = self._process_dict(input)

        # set initial state self.orbit[0]
        self._bootstrap(params)

        # set automata
        automaton = self._generate_automaton(params.dynamic_params)

        state = self.orbit[0]
        for step in range(self.final_step - 1):
            state = self._step(state, automaton)
            with torch.no_grad():
                self.orbit[step + 1] = state
                # if state is uniform, break
                # if torch.all(state == state[0]):
                #     break

    
        output_dict = deepcopy(input)
        # must detach here as gradients are not used
        # and this also leads to a deepcopy error downstream
        # also, squeezing leading dimensions for convenience
        output_dict["output"] = self.orbit[-1].detach().clone().squeeze()

        return output_dict

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
        for img in self.orbit:
            # need to squeeze leading dimensions
            parsed_img = img.squeeze().cpu().detach().numpy()
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
                byte_img, im_array, "mp4", fps=1, output_params=["-f", "mp4"]
            )
            return [(byte_img.getvalue(), "mp4")]
        else:
            raise NotImplementedError

    def _process_dict(self, input_dict: Dict) -> LeniaParameters:
        """
        Converts data_dictionary and parses for the correct
        parameters for Lenia.
        """
        init_params = deepcopy(input_dict["params"])
        if not isinstance(init_params, LeniaParameters):
            dyn_p = LeniaDynamicalParameters(**init_params["dynamic_params"])
            init_state = init_params["init_state"]
            params = LeniaParameters(dynamic_params=dyn_p, init_state=init_state)
        return params

    def _generate_automaton(self, dyn_params: LeniaDynamicalParameters) -> Any:
        automaton = LeniaStepFFT(
                SX=self.SX,
                SY=self.SY,
                R=torch.tensor(dyn_params.R),
                T=torch.tensor(dyn_params.T),
                m=torch.tensor(dyn_params.m),
                s=torch.tensor(dyn_params.s),
                b=dyn_params.b,
                kn=0,
                gn=1,
            )
        return automaton

    def _bootstrap(self, params: LeniaParameters):
        init_state = torch.zeros(
            1,
            1,
            self.SY,
            self.SX,
            dtype=torch.float64,
            requires_grad=False,
        )

        scaled_SY = int(self.SY / self.scale_init_state)
        scaled_SX = int( self.SX / self.scale_init_state)

        init_state[0, 0][
            self.SY // 2
            - math.ceil(scaled_SY / 2) : self.SY // 2
            + scaled_SY // 2,
            self.SX // 2
            - math.ceil(scaled_SX / 2) : self.SX // 2
            + scaled_SX // 2,
        ] = params.init_state.squeeze()
        # state is fixed deterministically by CPPN params,
        # so no need to save it after this point
        del params.init_state
        with torch.no_grad():
            self.orbit[0] = init_state


        return

    def _step(
        self, state: torch.Tensor, automaton: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        return automaton(state)


""" =============================================================================================
Lenia Main
============================================================================================= """


def create_colormap(colors: ndarray, is_marker_w: bool = True) -> List[int]:
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


def im_from_array_with_colormap(np_array: ndarray, colormap: List[int]) -> Image:
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


""" =============================================================================================
Lenia Main
============================================================================================= """

# Lenia family of functions for the kernel K and for the growth mapping g
kernel_core = {
    0: lambda r: (4 * r * (1 - r)) ** 4,  # polynomial (quad4)
    # exponential / gaussian bump (bump4)
    1: lambda r: torch.exp(4 - 1 / (r * (1 - r))),
    # step (stpz1/4)
    2: lambda r, q=1 / 4: (r >= q).double() * (r <= 1 - q).double(),
    # staircase (life)
    3: lambda r, q=1 / 4: (r >= q).double() * (r <= 1 - q).double()
    + (r < q).double() * 0.5,
}
field_func = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s**2))
    ** 4
    * 2
    - 1,
    # polynomial (quad4)
    # exponential / gaussian (gaus)
    1: lambda n, m, s: torch.exp(-((n - m) ** 2) / (2 * s**2)) * 2 - 1,
    2: lambda n, m, s: (torch.abs(n - m) <= s).double() * 2 - 1,  # step (stpz)
}


# Lenia Step FFT version (faster)
class LeniaStepFFT(torch.nn.Module):
    """Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        b: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        kn: int,
        gn: int,
        is_soft_clip: bool = False,
        SX: int = 256,
        SY: int = 256,
        device: str = "cpu",
    ) -> None:
        torch.nn.Module.__init__(self)

        self.register_buffer("R", R + 2)
        self.register_parameter("T", torch.nn.Parameter(T))
        self.register_buffer("b", b)
        self.register_parameter("m", torch.nn.Parameter(m))
        self.register_parameter("s", torch.nn.Parameter(s))

        self.kn = 0
        self.gn = 1

        self.SX = SX
        self.SY = SY
        self.spheric_pad = SphericPad(self.R)
        self.is_soft_clip = is_soft_clip

        self.device = device

        self.compute_kernel()

    def compute_kernel(self) -> None:
        # implementation of meshgrid in torch
        x = torch.arange(self.SX)
        y = torch.arange(self.SY)
        xx = x.repeat(self.SY, 1)
        yy = y.view(-1, 1).repeat(1, self.SX)

        X = (xx - int(self.SX / 2)).double() / float(self.SX / 2)
        Y = (yy - int(self.SY / 2)).double() / float(self.SY / 2)

        # canonical implementation from Mayalen, but I think there is a bug
        # X = (xx - int(self.SX / 2)).double() / float(self.R / 2)
        # Y = (yy - int(self.SY / 2)).double() / float(self.R / 2)

        # distance to center in normalized space
        D = torch.sqrt(X**2 + Y**2)

        # kernel
        k = len(self.b)  # modification to allow b always of length 4
        kr = k * D
        b = self.b[
            torch.min(torch.floor(kr).long(), (k - 1) * torch.ones_like(kr).long())
        ]
        kfunc = kernel_core[self.kn]
        kernel = (D < 1).double() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        self.kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
        # fft of the kernel
        self.kernel_FFT = torch.rfft(
            self.kernel_norm, signal_ndim=2, onesided=False
        ).to(self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        world_FFT = torch.rfft(input, signal_ndim=2, onesided=False)
        potential_FFT = complex_mult_torch(self.kernel_FFT, world_FFT)
        potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)
        potential = roll_n(potential, 3, potential.detach().size(3) // 2)
        potential = roll_n(potential, 2, potential.detach().size(2) // 2)

        gfunc = field_func[self.gn]
        field = gfunc(potential, self.m, self.s)

        if not self.is_soft_clip:
            output_img = torch.clamp(input + (1.0 / self.T) * field, min=0.0, max=1.0)
        else:
            output_img = soft_clip(input + (1.0 / self.T) * field, 0, 1, self.T)

        # if torch.any(torch.isnan(potential)):
        #     print("break")

        return output_img



