from copy import deepcopy
import torch
from examples.particlelenia.systems.ParticleLenia import ParticleLenia
from adtool.systems.System import System
from typing import Dict, Tuple, Optional
from pydantic import BaseModel
from pydantic.fields import Field
from adtool.utils.expose_config.expose_config import expose

# Define the configuration class for ParticleLeniaNoise
class ParticleLeniaNoiseConfig(BaseModel):
    n_particles: int = Field(400, ge=1, description="Number of particles")
    n_steps: int = Field(1000, ge=1, description="Number of steps to simulate")
    position_range: Tuple[float, float] = Field((-22.0, 22.0), description="Range for particle positions (x and y)")
    init_velocities: bool = Field(False, description="Whether to initialize velocities randomly")
    velocity_range: Tuple[float, float] = Field((0.0, 1.0), description="Range for particle velocities if init_velocities is True")

@expose
class ParticleLeniaNoise(ParticleLenia):
    config = ParticleLeniaNoiseConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_particles = self.config.n_particles
        self.position_range = self.config.position_range
        self.init_velocities = self.config.init_velocities
        self.velocity_range = self.config.velocity_range

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        # Generate random particle positions within the specified range
        positions = torch.rand((self.n_particles, 2)) * (self.position_range[1] - self.position_range[0]) + self.position_range[0]
        
        if self.init_velocities:
            velocities = torch.rand((self.n_particles, 2)) * (self.velocity_range[1] - self.velocity_range[0]) + self.velocity_range[0]
            intermed_dict['params']["init_state"] = torch.cat([positions, velocities], dim=-1)
        else:
            intermed_dict['params']["init_state"] = positions
        
        # Pass params to ParticleLenia
        intermed_dict = super().map(intermed_dict)
        return intermed_dict

    def render(self, data_dict, mode: str = "PIL_image") -> Tuple[bytes, str]:
        return super().render(data_dict, mode=mode)
