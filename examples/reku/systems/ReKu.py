import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import imageio
import io
import matplotlib.pyplot as plt

@dataclass
class ReKuParams:
 #   omega: np.ndarray  # Angular speeds (N,)
    initial_phases: np.ndarray  # Initial phases (N,)
    coupling_factor: float = 1.0  # Coupling factor



class ReKuSimulation:
    def __init__(
        self,
        N: int,
        noise_std: float = 0.01,
        dt: float = 0.01,
        total_t: float = 10.0,
        n_skr: int = 10,
        device: str = 'cpu'
    ) -> None:
        self.N = N  # Number of populations
        self.noise_std = noise_std
        self.dt = dt
        self.total_t = total_t
        self.n_skr = n_skr
        self.device = device
        self.phi = None  # Phases
        self.omega = None  # Angular speeds
        self.phi_history = []  # History of phases


    def update(self, coupling_fct):
        """Integrate the ReKu model using Euler method with Gaussian noise"""
        times = np.arange(0., self.total_t, self.dt)
        counter = 0

        for t in times[1:]:
            dphi = np.zeros(self.N)

            # Calculate phase changes with angular speed and coupling effect
            for i in range(self.N):
                dphi[i] = self.omega[i]  # Angular speed contribution

                # Add the coupling effect from other oscillators
                for j in range(self.N):
                    if i != j:
                        dphi[i] += coupling_fct(self.phi[i], self.phi[j]) * self.params.coupling_factor

            # Euler update with noise
            self.phi += dphi * self.dt
            self.phi += np.random.normal(loc=0., scale=self.noise_std, size=self.N) * np.sqrt(self.dt)

            counter += 1
            if t>9*self.N/10 and counter % self.n_skr == 0:
                self.phi_history.append(self.phi.copy())

    def map(self, input: Dict) -> Dict:
        # Update to focus on angular speeds and phases
        self.params = ReKuParams(
      #      omega=np.array(input["params"]["dynamic_params"]["omega"]),
            initial_phases=np.array(input["params"]["dynamic_params"]["initial_phases"]),
            coupling_factor=input["params"]["dynamic_params"]["coupling_factor"]
        )



        # Randomly initialize phases and angular speeds
        self.phi = self.params.initial_phases.copy()
    #    self.omega = self.params.omega.copy()
        self.omega=np.ones(self.N)
        self.phi_history = [self.phi.copy()]


        self.update(coupling_fct=self.coupling_fct)
        input["output"] = self.phi_history
        return input

    def coupling_fct(self, phi_1, phi_2):
        """Coupling function based on phase difference"""
        dphi = phi_2 - phi_1
        return np.sin(dphi) if dphi > 0 else 0

    def _render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        phi_history = data_dict["output"]
        num_frames = len(phi_history)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')

        def plot_frame(frame_idx):
            ax.clear()
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            ax.axis('off')
            phi = phi_history[frame_idx]

            x = np.cos(phi)
            y = np.sin(phi)

            ax.scatter(x, y, s=10)

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
        return [(byte_img.getvalue(), "mp4")]
    

    # render juste a single plot with time as x-axis and phase as y-axis
    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:

        # compute the derivative of each phase
        dphi = np.diff(data_dict["output"], axis=0)
        # sum it's absolute value to get a scalar value
        dphi = np.sum(np.abs(dphi), axis=1)
        # find the maximum value index
        max_idx = np.argmax(dphi)
        # define an interval around the maximum value, with respect to the total number of frames
        nb_visu_frames=400
        start_idx = max(0, max_idx - nb_visu_frames // 2)
        end_idx = min(len(data_dict["output"]), max_idx + nb_visu_frames // 2)

        start_idx=0
        end_idx=len(data_dict["output"])

        # return a png
        fig, ax = plt.subplots()
      #  ax.plot(data_dict["output"])
      # don't plot the phase directly, but the sin of the phase
        ax.plot(np.sin(data_dict["output"][start_idx:end_idx]
                       ))
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Phase")
        # hide the axes
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return [(buf.getvalue(), "png")]




class GenerationParams(BaseModel):
    N: int = Field(2, ge=1, le=100)
    noise_std: float = Field(0.01, ge=0.0, le=1.0)
    dt: float = Field(0.01, ge=0.001, le=1.0)
    total_t: float = Field(10.0, ge=0.1, le=10000.0)

from adtool.utils.expose_config.expose_config import expose

@expose
class ReKu(ReKuSimulation):
    config = GenerationParams

    def __init__(
        self,
        N: int,
        noise_std: float=0.01,
        dt: float=0.01,
        total_t: float=10.0,
        device: str = 'cpu'
    ) -> None:
        super().__init__(
            N=N,
            noise_std=noise_std,
            dt=dt,
            total_t=total_t,
            device=device
        )
