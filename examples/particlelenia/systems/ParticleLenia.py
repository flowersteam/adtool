from examples.particlelenia.systems.ParticleLeniaParameters import ParticleLeniaDynamicalParameters, ParticleLeniaParameters
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import io
from dataclasses import dataclass
from collections import namedtuple
from copy import deepcopy

from typing import Any, Dict, Tuple

import imageio


# Utility functions
def norm(v, axis=-1, keepdims=False, eps=0.0):
    return torch.sqrt(torch.clamp((v*v).sum(axis=axis, keepdim=keepdims), min=eps))

def normalize(v, axis=-1, eps=1e-20):
    return v / norm(v, axis=axis, keepdims=True, eps=eps)

def peak_f(x, mu, sigma, a):
    return a * torch.exp(-((x - mu) / sigma) ** 2)

# Named tuples for parameters and fields
Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')

class ParticleLeniaSystem:
    def __init__(self, 
                 mu_k=2.75, 
                 sigma_k=1.25, 
                 w_k=0.02625,
                    mu_g=0.7,
                    sigma_g=0.16666666666666669,
                    c_rep=1.0,

                 initial_coordinates = torch.rand((400, 2)) * 12.0 - 6.0
                 ):
        
        self.params = Params(mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep)

        self.points0 = initial_coordinates
        self.history = [initial_coordinates]

    def fields_f(self, points, x):
        # Ensure points are PyTorch tensors
        if isinstance(points, np.ndarray):
            points = torch.tensor(points, dtype=torch.float32)
        
        # Reshape x and points to broadcast properly
        x_expanded = x.unsqueeze(1)  # Shape: (640000, 1, 2)
        points_expanded = points.unsqueeze(0)  # Shape: (1, 400, 2)
        
        # Calculate pairwise distances
        r = torch.sqrt(torch.clamp(torch.sum((x_expanded - points_expanded) ** 2, dim=-1), min=1e-10))  # Shape: (640000, 400)
        
        # Summing across the particles to get the field at each grid point
        U = torch.sum(peak_f(r, self.params.mu_k, self.params.sigma_k, self.params.w_k), dim=-1)  # Shape: (640000,)
        G = peak_f(U, self.params.mu_g, self.params.sigma_g, 1.0)
        R = torch.sum(self.params.c_rep / 2 * torch.clamp(1.0 - r, min=0.0) ** 2, dim=-1)  # Shape: (640000,)
        
        return Fields(U, G, R, E=R - G)


    def field_x(self, points):
        return lambda x: self.fields_f(points, x)

    def motion_f(self, points):
        def grad_E(x):
            # Compute the energy field E at x
            x.requires_grad_(True)
            energy = self.fields_f(points, x).E
            total_energy = torch.sum(energy)  # Sum to get a scalar value
            grad = torch.autograd.grad(total_energy, x, create_graph=True)[0]
            x.requires_grad_(False)
            return grad
        
        return grad_E

    def odeint_euler(self, dt, n):
        def step_f(x):
            grad_E = self.motion_f(x)(x)
            with torch.no_grad():
                x = x - dt * grad_E
            return x
        
     #   print(f"Starting simulation with params {self.params}")
        for i in range(1, n):
            new_points = step_f(self.history[-1])
            self.history.append(new_points)
            # if i % 2000 == 0:
            #     print(f"Step {i}")

        return self.history

class ParticleLenia:
    def __init__(self, n_particles=400, dt=0.1, n_steps=10000, **kwargs):
        self.dt = dt
        self.n_steps = n_steps
        self.n_particles = n_particles

        # Initialize particles randomly in the specified range
        self.points0 = torch.rand((n_particles, 2)) * 12.0 - 6.0

    def _process_dict(self, input_dict: Dict) -> ParticleLeniaParameters:
        """
        Converts data_dictionary and parses for the correct
        parameters for Lenia.
        """
       
        init_params = deepcopy(input_dict["params"])
        if not isinstance(init_params, ParticleLeniaParameters):
            dyn_p = ParticleLeniaDynamicalParameters(**init_params["dynamic_params"])
            init_state = init_params["init_state"]
            params = ParticleLeniaParameters(dynamic_params=dyn_p, init_state=init_state)
        return params

    def _generate_automaton(self, dyn_params: ParticleLeniaDynamicalParameters) -> Any:
        automaton = ParticleLeniaSystem(
                mu_k=dyn_params.mu_k,
                sigma_k=dyn_params.sigma_k,
                w_k=dyn_params.w_k,
                mu_g=dyn_params.mu_g,
                sigma_g=dyn_params.sigma_g,
                c_rep=dyn_params.c_rep,
                initial_coordinates=self.points0
            )
        return automaton
    

    def map(self, input_dict: Dict) -> Dict:
        params = params = self._process_dict(input_dict)

        particle_lenia = self._generate_automaton(params.dynamic_params)

        

        # Evolve the system
        self.history=particle_lenia.odeint_euler(self.dt, self.n_steps)
        
        output_dict = deepcopy(input_dict)
        output_dict["output"] = particle_lenia.history[-1]
        return output_dict

    def render(self, data_dict, mode: str = "matplotlib") -> Tuple[bytes, str]:
        # im_array = [frame for frame in self.history]
        
        # byte_img = io.BytesIO()
        # imageio.mimwrite(byte_img, im_array, format='mp4', fps=self.n_steps//10)
        # byte_img.seek(0)
        # return byte_img.getvalue(), "mp4"

        #this is a list of coordinates (x,y) for each particle, at each time step
        # create a video of the particle trajectories mp4
        # first convert the list of coordinates to a list of images

        # Create a list of images
        images = []
        for i in range(0, self.n_steps, self.n_steps//100):
            fig, ax = plt.subplots()

            # same but take the full image
            ax.scatter(self.history[i][:, 0], self.history[i][:, 1], s=30, c="black")

            # Set limits and aspect ratio
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_aspect('equal')

            # Remove margins
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # hide axes
            ax.axis('off')

            

            # Save to byte buffer
            byte_img = io.BytesIO()
            plt.savefig(byte_img, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            byte_img.seek(0)
            images.append(imageio.imread(byte_img))


            
        # Create the video
        byte_img = io.BytesIO()
        imageio.mimsave(byte_img, images, format='mp4', fps=10)
        byte_img.seek(0)
        return byte_img.getvalue(), "mp4"

# # Example usage
# dt = 0.1
# particle_lenia_system = ParticleLeniaSystem( n_particles=400, dt=dt, n_steps=1000)

# # Initial state
# init_state = {
#     "params": {
#         "dynamic_params": {
#             "mu_k": 2.75,
#             "sigma_k": 1.25,
#             "w_k": 0.02625,
#             "mu_g": 0.7,
#             "sigma_g": 0.16666666666666669,
#             "c_rep": 1.0,
#         },
#         "init_state": None
#     }
# }   

# # Mapping the initial state
# output_dict = particle_lenia_system.map(init_state)



# # Rendering the final state
# particle_lenia_system.render(output_dict, mode="matplotlib")
