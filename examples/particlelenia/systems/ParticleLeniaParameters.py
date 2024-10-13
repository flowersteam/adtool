import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Tuple, Union
import torch
import numpy as np

@dataclass
class ParticleLeniaDynamicalParameters:
    mu_k: float = 2.75
    sigma_k: float = 1.25
    w_k: float = 0.02625
    mu_g: float = 0.7
    sigma_g: float = 0.16666666666666669
    c_rep: float = 1.0

    def __post_init__(self):
        # Convert out of tensors if necessary
        if isinstance(self.mu_k, torch.Tensor):
            self.mu_k = self.mu_k.item()
        if isinstance(self.sigma_k, torch.Tensor):
            self.sigma_k = self.sigma_k.item()
        if isinstance(self.w_k, torch.Tensor):
            self.w_k = self.w_k.item()
        if isinstance(self.mu_g, torch.Tensor):
            self.mu_g = self.mu_g.item()
        if isinstance(self.sigma_g, torch.Tensor):
            self.sigma_g = self.sigma_g.item()
        if isinstance(self.c_rep, torch.Tensor):
            self.c_rep = self.c_rep.item()

        # Ensure parameters are within constraints
        self.mu_k = min(5.0, max(0.0, self.mu_k))
        self.sigma_k = min(2.0, max(0.1, self.sigma_k))
        self.w_k = min(0.1, max(0.0, self.w_k))
        self.mu_g = min(1.0, max(0.0, self.mu_g))
        self.sigma_g = min(0.5, max(0.05, self.sigma_g))
        self.c_rep = min(2.0, max(0.0, self.c_rep))

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.mu_k, self.sigma_k, self.w_k, self.mu_g, self.sigma_g, self.c_rep])
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.mu_k, self.sigma_k, self.w_k, self.mu_g, self.sigma_g, self.c_rep])

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        mu_k = tensor[0].item()
        sigma_k = tensor[1].item()
        w_k = tensor[2].item()
        mu_g = tensor[3].item()
        sigma_g = tensor[4].item()
        c_rep = tensor[5].item()
        return cls(mu_k=mu_k, sigma_k=sigma_k, w_k=w_k, mu_g=mu_g, sigma_g=sigma_g, c_rep=c_rep)


@dataclass
class ParticleLeniaParameters:
    """Holds input parameters for ParticleLenia model."""

    dynamic_params: ParticleLeniaDynamicalParameters = field(default_factory=lambda: ParticleLeniaDynamicalParameters())
    init_state: torch.Tensor = field(default_factory=lambda: torch.rand((400, 2)) * 12.0 - 6.0)


@dataclass
class ParticleLeniaHyperParameters:
    """Holds parameters to initialize ParticleLenia model."""

    tensor_low: torch.Tensor = ParticleLeniaDynamicalParameters().to_tensor()
    tensor_high: torch.Tensor = ParticleLeniaDynamicalParameters().to_tensor()
    tensor_bound_low: torch.Tensor = ParticleLeniaDynamicalParameters(
        mu_k=0.0,
        sigma_k=0.1,
        w_k=0.0,
        mu_g=0.0,
        sigma_g=0.05,
        c_rep=0.0
    ).to_tensor()
    tensor_bound_high: torch.Tensor = ParticleLeniaDynamicalParameters(
        mu_k=5.0,
        sigma_k=2.0,
        w_k=0.1,
        mu_g=1.0,
        sigma_g=0.5,
        c_rep=2.0
    ).to_tensor()

    init_state_dim: Tuple[int, int] = (400, 2)
