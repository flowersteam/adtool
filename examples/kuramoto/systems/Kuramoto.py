import numpy as np
import torch
from torch import nn
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import imageio
import io
import matplotlib.pyplot as plt

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class KuramotoParams:
    intra_couplings: np.ndarray  # Intra-population coupling constants (N,)
    inter_couplings: np.ndarray  # Inter-population coupling constants (N, N)

    def to_tensor(self):
        intra_tensor = torch.tensor(self.intra_couplings, dtype=torch.float32)
        inter_tensor = torch.tensor(self.inter_couplings, dtype=torch.float32)
        return intra_tensor, inter_tensor

    @classmethod
    def from_tensor(cls, intra_tensor, inter_tensor):
        return cls(
            intra_couplings=intra_tensor.numpy(),
            inter_couplings=inter_tensor.numpy()
        )

class KuramotoSimulation:
    def __init__(
        self,
        N: int,
        K: int,
        frequency_min: float = 0,
        frequency_max:  float = 0.05,
        num_inference_steps: int = 1000,
        initial_condition_seed: int = 0,
        device: str = 'cpu'
    ) -> None:
        self.N = N  # Number of populations
        self.K = K  # Number of oscillators per population
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.num_inference_steps = num_inference_steps
        self.initial_condition_seed = initial_condition_seed
        self.device = device
        self.theta = None  # Oscillator phases
        self.params = None
        self.natural_frequencies = None
        self.theta_history = []  # History of theta values

    def _initialize_conditions(self):
        np.random.seed(self.initial_condition_seed)
        self.theta = np.random.uniform(0, 2 * np.pi, (self.N, self.K))
        self.natural_frequencies = np.random.uniform(self.frequency_min, self.frequency_max, (self.N, self.K))
        self.theta_history = [self.theta.copy()]

    def update(self):
        """Integrate the Kuramoto model using the Euler method"""
        for _ in range(self.num_inference_steps):
            dtheta = np.zeros((self.N, self.K))


            for i in range(self.N):
                for j in range(self.K):
                    dtheta[i, j] = self.natural_frequencies[i, j]
                    
                    for k in range(self.N):
                        for l in range(self.K):
                            if i != k:
                                dtheta[i, j] += self.params.inter_couplings[i, k] * np.sin(self.theta[k, l] - self.theta[i, j])/self.K
                            else:
                                dtheta[i, j] += self.params.intra_couplings[i] * np.sin(self.theta[k, l] - self.theta[i, j])/self.K
                            


            self.theta += dtheta


            self.theta = np.mod(self.theta, 2 * np.pi)



            self.theta_history.append(self.theta.copy())

    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        self.params = KuramotoParams(
            intra_couplings=np.array(input["params"]["dynamic_params"]["intra_couplings"]),
            inter_couplings=np.array(input["params"]["dynamic_params"]["inter_couplings"])
        )

        if fix_seed:
            np.random.seed(self.initial_condition_seed)
        self._initialize_conditions()
        self.update()
        input["output"] = self.theta_history
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        theta_history = data_dict["output"]
        num_frames = len(theta_history)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
       # circle = plt.Circle((0, 0), 1, color='black', fill=False)
       # ax.add_artist(circle)

        def plot_frame(frame_idx):
            ax.clear()
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            ax.axis('off')  # Hide the axis
      #      ax.add_artist(circle)
            theta = theta_history[frame_idx]
            
            x = np.cos(theta)
            y = np.sin(theta)
            # x.shape = (N, K)
            # y.shape = (N, K)
            for i in range(self.N):
                ax.scatter(x[i], y[i], s=10)
          #  ax.legend()



        frames = []
        for i in range(0, num_frames, 1):
            plot_frame(i)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()

        byte_img = io.BytesIO()
        imageio.mimwrite(byte_img, frames, format='mp4', fps=10)
        byte_img.seek(0)
        return byte_img.getvalue(), "mp4"

class GenerationParams(BaseModel):
    N: int = Field(2, ge=1, le=100)
    K: int = Field(10, ge=1, le=1000)
    num_inference_steps: int = Field(1000, ge=1, le=10000)

from adtool.utils.expose_config.expose_config import expose

@expose
class Kuramoto(KuramotoSimulation):
    config = GenerationParams

    def __init__(
        self,
        N: int,
        K: int,
        num_inference_steps: int,
        device: str = 'cpu'
    ) -> None:
        super().__init__(
            N=N,
            K=K,
            num_inference_steps=num_inference_steps,
            device=device
        )