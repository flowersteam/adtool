import dataclasses
import io
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

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





from examples.flowlenia.systems.FlowLeniaParameters import FlowLeniaDynamicalParameters, FlowLeniaKernelGrowthDynamicalParameters,FlowLeniaParameters




from pydantic import BaseModel
from pydantic.fields import Field
from adtool.utils.expose_config.expose_config import expose



from adtool.systems.System import System
from examples.flowlenia.systems.ReintegrationTracking import ReintegrationTracking
from examples.flowlenia.systems.Utils import conn_from_matrix, growth, ker_f, sigmoid, sobel


torch.set_default_dtype(torch.float32)

class FlowLenia(System):


    def __init__(self, SX=256,
                 SY=256,
                 dt=.2,
                 dd=5,
                 sigma=.65,
                 final_step=200,
                 scale_init_state=1,


                     nb_k: int = 10,
    C: int = 1,
    n: int = 2, #exponent for alpha
    theta_A : float = 1.,
    R:int=10,
                **kwargs
                 
                 ):
        super().__init__()
        self.SX= SX
        self.SY= SY
        self.dt= dt
        self.dd= dd
        self.sigma= sigma
        self.scale_init_state= scale_init_state
        self.final_step= final_step
        self.nb_k= nb_k
        self.C= C
        self.n= n
        self.R= R
        self.theta_A= theta_A


        self.locator = BlobLocator()
        self.orbit = torch.empty(
                (self.final_step, self.SX, self.SY,self.C),
            requires_grad=False,
        )

        M = torch.ones((C, C), dtype=int) * self.nb_k

        self.nb_k = int(M.sum())

        self.c0, self.c1 = conn_from_matrix(M)




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

        output_dict = deepcopy(input)
        # must detach here as gradients are not used
        # and this also leads to a deepcopy error downstream
        # also, squeezing leading dimensions for convenience
        output_dict["output"] = self.orbit[-1].detach().clone()

        return output_dict
    


    def render(self, data_dict, mode: str = "PIL_image") -> Optional[bytes]:
        # ignores data_dict, as the render is based on self.orbit
        # in which only the last state is stored in data_dict["output"]


        im_array = []
        for img in self.orbit:
            # need to squeeze leading dimensions
            parsed_img = img.cpu().detach().numpy()

            img=state2img(parsed_img)
            if img.dtype in [np.float32, np.float64]:
                img = np.uint8(img.clip(0, 1)*255)
            if len(img.shape) == 2:
                img = np.repeat(img[..., None], 3, -1)
            im_array.append(img)

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
                byte_img, im_array, "mp4", fps=10, output_params=["-f", "mp4"]
            )
            return byte_img.getvalue()
        else:
            raise NotImplementedError

    def _process_dict(self, input_dict: Dict) -> FlowLeniaParameters:
        """
        Converts data_dictionary and parses for the correct
        parameters for Lenia.
        """
        init_params = deepcopy(input_dict["params"])
        if not isinstance(init_params, FlowLeniaParameters):
            dyn_p = FlowLeniaDynamicalParameters(**init_params["dynamic_params"])
            init_state = init_params["init_state"]
            params = FlowLeniaParameters(dynamic_params=dyn_p, init_state=init_state)
        return params

    def _generate_automaton(self, dyn_params: FlowLeniaDynamicalParameters) -> Any:  
        
        automaton = TorchFlowLenia(
                SX=self.SX,
                SY=self.SY,
                dt=self.dt,
                dd=self.dd,
                sigma=self.sigma,
                final_step=self.final_step,
                scale_init_state=self.scale_init_state,
                C=self.C,
                n=self.n,
                theta_A=self.theta_A,
                R=dyn_params.R,
                c0=self.c0,
                c1=self.c1,
                kernelgrowths=dyn_params.KernelGrowths,
            )
        return automaton

    def _bootstrap(self, params: FlowLeniaParameters):

        init_state = torch.zeros(
            self.SY,
            self.SX,
            self.C,
            dtype=torch.float64,
            requires_grad=False,
        )

        scaled_SY = int(self.SY / self.scale_init_state)
        scaled_SX = int( self.SX / self.scale_init_state)



        init_state[
            self.SY // 2
            - math.ceil(scaled_SY / 2) : self.SY // 2
            + scaled_SY // 2,
            self.SX // 2
            - math.ceil(scaled_SX / 2) : self.SX // 2
            + scaled_SX // 2,
            
        ] = params.init_state
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

def state2img(A):
    C = A.shape[-1]
    if C == 1:
        return A[..., 0]
    if C == 2:
        return np.dstack([A[..., 0], A[..., 0], A[..., 1]])
    return A[..., :3]



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
class TorchFlowLenia(torch.nn.Module):
    """Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(
        self,

                SX=256,
                SY=256,
                dt=0.2,
                dd=5,
                sigma=0.65,
                final_step=200,
                scale_init_state=1,
                C: int = 1,
                n: int = 2,
                theta_A: float = 1.0,
                R: int = 10,
                c0: List[List[int]] = [[0]],
                c1: List[List[int]] = [[0]],
                kernelgrowths: List[FlowLeniaKernelGrowthDynamicalParameters] = None,

        device: str = "cpu",
    ) -> None:
        torch.nn.Module.__init__(self)

        self.register_buffer("dt", torch.tensor(dt))
        self.register_buffer("dd", torch.tensor(dd))
        self.register_buffer("sigma", torch.tensor(sigma))
        self.register_buffer("final_step", torch.tensor(final_step))
        self.register_buffer("scale_init_state", torch.tensor(scale_init_state))
        self.register_buffer("C", torch.tensor(C))
        self.register_buffer("n", torch.tensor(n))
        self.register_buffer("theta_A", torch.tensor(theta_A))

        self.R = R
        self.kernelgrowths = kernelgrowths


        self.c0 = c0
        self.c1 = c1



        self.SX = SX
        self.SY = SY
        self.spheric_pad = SphericPad(self.R)

        self.RT = ReintegrationTracking(self.SX, self.SY, self.dt,
            self.dd, self.sigma)

        self.device = device



        self.compute_kernel()

    def compute_kernel(self) -> None:
        """Compute kernels and return a dic containing kernels fft

        Args:
            params (Params): raw params of the system

        Returns:
            CompiledParams: compiled params which can be used as update rule
        """
        midX=self.SX//2
        midY=self.SY//2
        

        # Ds = [ np.linalg.norm(np.mgrid[-midX:midX, -midY:midY], axis=0) /
        #         ((self.R+15) * k.r) for k in self.kernelgrowths ]  # (x,y,k)
        

        

        x = torch.linspace(-midX, midX, 2*midX, dtype=torch.float)
        y = torch.linspace(-midY, midY, 2*midY, dtype=torch.float)
        X, Y = torch.meshgrid(x, y)
        Ds = [ torch.norm(torch.stack((X, Y), dim=0), dim=0) /
                ((self.R+15) * k.r) for k in self.kernelgrowths ]



        
        



        K = torch.stack([sigmoid(-(D-1)*10) * ker_f(D, k.a, k.w, k.b)
                        for k, D in zip(self.kernelgrowths, Ds)])
        
        K=K.transpose(0,2)
    

        nK = K / torch.sum(K, dim=(0,1), keepdims=True)
    
        fK = torch.fft.fft2(torch.fft.fftshift(nK, dim=(0,1)), dim=(0,1))


        self.fK=fK
        self.K=K
        self.nK=nK

        self.m = torch.tensor([k.m for k in self.kernelgrowths], device=self.device)
        self.s = torch.tensor([k.s for k in self.kernelgrowths], device=self.device)
        self.h = torch.tensor([k.h for k in self.kernelgrowths], device=self.device)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # A has shape [1, 1, 256, 256, 2]


        fA = torch.fft.fft2(A,dim=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.c0]  # (x,y,k)


        U = torch.fft.ifft2(self.fK * fAk,dim=(0,1) ).real  # (x,y,k)
        
        U = growth(U, self.m, self.s) * self.h  # (x,y,k)


        U = torch.stack([ U[:, :, self.c1[c]].sum(axis=-1) for c in range(self.C) ], dim=-1)  # (x,y,c)

        #-------------------------------FLOW------------------------------------------
        nabla_U = sobel(U) #(x, y, 2, c)   

        nabla_A = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)

        alpha = torch.clip((A[:, :, None, :]/self.theta_A)**self.n, .0, 1.)

        F = nabla_U * (1 - alpha)  - nabla_A * alpha #

        nA = self.RT(A, F)



        return nA