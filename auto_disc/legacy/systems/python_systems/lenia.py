import io
import math
import typing

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from addict import Dict
from auto_disc.legacy.systems.python_systems import BasePythonSystem

from auto_disc.legacy.utils.misc.torch_utils import (
    SphericPad,
    complex_mult_torch,
    roll_n,
    soft_clip,
)
from auto_disc.legacy.utils.mutators import GaussianMutator
from auto_disc.legacy.utils.spaces import BoxSpace, DictSpace, DiscreteSpace
from auto_disc.legacy.utils.spaces.utils import ConfigParameterBinding
from matplotlib.animation import FuncAnimation
from numpy import ndarray
from PIL import Image

""" =============================================================================================
System definition
============================================================================================= """


# @StringConfigParameter(
#     name="version",
#     possible_values=["pytorch_fft", "pytorch_conv2d"],
#     default="pytorch_fft",
# )
# @IntegerConfigParameter(name="SX", default=256, min=1)
# @IntegerConfigParameter(name="SY", default=256, min=1)
# @IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
# @IntegerConfigParameter(name="scale_init_state", default=1, min=1)

from auto_disc.auto_disc.utils.expose_config.defaults import Defaults, defaults
from dataclasses import dataclass, field

@dataclass
class LeniaConfig(Defaults):
    version: str=defaults("pytorch_fft", domain=["pytorch_fft", "pytorch_conv2d"])
    SX: int=defaults(256, min=1)
    SY: int=defaults(256, min=1)
    final_step: int=defaults(200, min=1, max=1000)
    scale_init_state: int=defaults(1, min=1)

@LeniaConfig.expose_config()
class Lenia(BasePythonSystem):

    input_space = DictSpace(
        init_state=BoxSpace(
            low=0.0,
            high=1.0,
            mutator=GaussianMutator(mean=0.0, std=0.5),
            shape=(
                ConfigParameterBinding("SX")
                // ConfigParameterBinding("scale_init_state"),
                ConfigParameterBinding("SY")
                // ConfigParameterBinding("scale_init_state"),
            ),
        ),
        R=DiscreteSpace(n=20, mutator=GaussianMutator(mean=0.0, std=0.5), indpb=1.0),
        T=BoxSpace(
            low=1.0,
            high=20.0,
            mutator=GaussianMutator(mean=0.0, std=0.5),
            shape=(),
            indpb=1.0,
            dtype=torch.float32,
        ),
        b=BoxSpace(
            low=0.0,
            high=1.0,
            mutator=GaussianMutator(mean=0.0, std=0.1),
            shape=(4,),
            indpb=1.0,
            dtype=torch.float32,
        ),
        m=BoxSpace(
            low=0.0,
            high=1.0,
            mutator=GaussianMutator(mean=0.0, std=0.1),
            shape=(),
            indpb=1.0,
            dtype=torch.float32,
        ),
        s=BoxSpace(
            low=0.001,
            high=0.3,
            mutator=GaussianMutator(mean=0.0, std=0.05),
            shape=(),
            indpb=1.0,
            dtype=torch.float32,
        ),
        # kn = DiscreteSpace(n=4, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=1.0),
        # gn = DiscreteSpace(n=3, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=1.0),
    )

    output_space = DictSpace(
        states=BoxSpace(
            low=0,
            high=1,
            shape=(
                ConfigParameterBinding("final_step"),
                ConfigParameterBinding("SX"),
                ConfigParameterBinding("SY"),
            ),
        )
    )

    step_output_space = DictSpace(
        state=BoxSpace(
            low=0,
            high=1,
            shape=(ConfigParameterBinding("SX"), ConfigParameterBinding("SY")),
        )
    )

    def reset(
        self, run_parameters: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        run_parameters.kn = 0
        run_parameters.gn = 1
        init_state = torch.zeros(
            1, 1, self.config.SY, self.config.SX, dtype=torch.float64
        )
        init_state[
            0,
            0,
            self.config.SY // 2
            - math.ceil(self.input_space["init_state"].shape[0] / 2) : self.config.SY
            // 2
            + self.input_space["init_state"].shape[0] // 2,
            self.config.SX // 2
            - math.ceil(self.input_space["init_state"].shape[0] / 2) : self.config.SX
            // 2
            + self.input_space["init_state"].shape[0] // 2,
        ] = run_parameters.init_state
        # self.state = init_state.to(self.device)
        self._state = init_state
        del run_parameters.init_state

        if self.config.version.lower() == "pytorch_fft":
            self._automaton = LeniaStepFFT(
                SX=self.config.SX, SY=self.config.SY, **run_parameters
            )
        elif self.config.version.lower() == "pytorch_conv2d":
            self._automaton = LeniaStepConv2d(**run_parameters)
        else:
            raise ValueError(
                "Unknown lenia version (config.version = {!r})".format(
                    self.config.version
                )
            )

        self._observations = Dict()
        # self._observations.timepoints = list(range(self.config.final_step))
        self._observations.states = torch.empty(
            (self.config.final_step, self.config.SX, self.config.SY)
        )
        self._observations.states[0] = self._state

        self._step_idx = 0

        current_observation = Dict()
        current_observation.state = self._observations.states[0]

        return current_observation

    def step(
        self, action=None
    ) -> typing.Tuple[typing.Dict[str, torch.Tensor], int, bool, None]:
        if self._step_idx >= self.config.final_step:
            raise Exception("Final step already reached, please reset the system.")

        self._step_idx += 1
        self._state = self._automaton(self._state)

        self._observations.states[self._step_idx] = self._state[0, 0, :, :]

        current_observation = Dict()
        current_observation.state = self._observations.states[self._step_idx]

        return (
            current_observation,
            0,
            self._step_idx >= self.config.final_step - 1,
            None,
        )

    def observe(self) -> typing.Dict[str, torch.Tensor]:
        return self._observations

    def render(self, mode: str = "PIL_image") -> typing.Any:
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
        for img in self._observations.states:
            im = im_from_array_with_colormap(img.cpu().detach().numpy(), colormap)
            im_array.append(im.convert("RGB"))

        if mode == "human":
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
                byte_img, im_array, "mp4", fps=30, output_params=["-f", "mp4"]
            )
            return (byte_img, "mp4")
        else:
            raise NotImplementedError

    def close(self):
        pass


""" =============================================================================================
Lenia Main
============================================================================================= """


def create_colormap(colors: ndarray, is_marker_w: bool = True) -> typing.List[int]:
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


def im_from_array_with_colormap(np_array: ndarray, colormap: typing.List[int]) -> Image:
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
        X = (xx - int(self.SX / 2)).double() / float(self.R)
        Y = (yy - int(self.SY / 2)).double() / float(self.R)

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


# Lenia Step Conv2D version
class LeniaStepConv2d(torch.nn.Module):
    """Module pytorch that computes one Lenia Step with the conv2d version"""

    def __init__(self, R, T, b, m, s, kn, gn, is_soft_clip=False, device="cpu"):
        torch.nn.Module.__init__(self)

        self.register_buffer("R", R + 2)
        self.register_parameter("T", torch.nn.Parameter(T))
        self.register_buffer("b", b)
        self.register_parameter("m", torch.nn.Parameter(m))
        self.register_parameter("s", torch.nn.Parameter(s))

        self.kn = 0
        self.gn = 1

        self.spheric_pad = SphericPad(self.R)
        self.is_soft_clip = is_soft_clip

        self.device = device

        self.compute_kernel()

    def compute_kernel(self) -> None:
        SY = 2 * self.R + 1
        SX = 2 * self.R + 1

        # implementation of meshgrid in torch
        x = torch.arange(SX)
        y = torch.arange(SY)
        xx = x.repeat(int(SY.item()), 1)
        yy = y.view(-1, 1).repeat(1, int(SX.item()))
        X = (xx - int(SX / 2)).double() / float(self.R)
        Y = (yy - int(SY / 2)).double() / float(self.R)

        # distance to center in normalized space
        D = torch.sqrt(X**2 + Y**2)

        # kernel
        k = len(self.b)
        kr = k * D
        b = self.b[
            torch.min(torch.floor(kr).long(), (k - 1) * torch.ones_like(kr).long())
        ]
        kfunc = kernel_core[self.kn]
        kernel = (D < 1).double() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        self.kernel_norm = (
            (kernel / kernel_sum).unsqueeze(0).unsqueeze(0).to(self.device)
        )

    def forward(self, input):
        potential = torch.nn.functional.conv2d(
            self.spheric_pad(input), weight=self.kernel_norm
        )
        gfunc = field_func[self.gn]
        field = gfunc(potential, self.m, self.s)

        if not self.is_soft_clip:
            output_img = torch.clamp(input + (1.0 / self.T) * field, min=0.0, max=1.0)
        else:
            output_img = soft_clip(input + (1.0 / self.T) * field, 0, 1, self.T)

        return output_img
