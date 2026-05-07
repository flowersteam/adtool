from copy import deepcopy
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field

from adtool.systems import System
from adtool.utils.expose_config.expose_config import expose
from examples.embedded_systems.explorers.embedded_systems_explorer import (
    BaseExplorerFactory,
    BaseIMGEPInstance,
)
from examples.embedded_systems.examples.core_interferences.types import (
    InstructionProgram,
    InterferenceParamsPayload,
)
from examples.embedded_systems.helpers.module_factory import make_module


class InterferenceIMGEPConfig(BaseModel):
    periode: int = Field(1, ge=1, le=100000)
    k: int = Field(1, ge=1, le=1000)
    mixer: str = Field(
        "examples.embedded_systems.examples.core_interferences.mixers.interference_chunk_mixer.ChunkProgramMixer"
    )
    mixer_config: Dict = Field(default_factory=lambda: {"num_parts": 2})
    behavior_map_config: Dict = Field(default_factory=lambda: {
        "path": "examples.embedded_systems.examples.core_interferences.maps.InterferenceBehaviorMap.InterferenceBehaviorMap"
    })
    parameter_map_config: Dict = Field(default_factory=lambda: {
        "path": "examples.embedded_systems.examples.core_interferences.maps.InterferenceParameterMap.InterferenceParameterMap"
    })


class InterferenceIMGEPInstance(BaseIMGEPInstance):
    """Interference-specific IMGEP policy with kNN+mix behavior."""

    def _compose_base_policy(
        self,
        selected_params: List[InterferenceParamsPayload],
    ) -> InterferenceParamsPayload:
        if not selected_params:
            return self.parameter_map.sample()

        if len(selected_params) == 1 or self.k <= 1:
            return deepcopy(selected_params[0])

        core0_pool = [p["dynamic_params"]["core0"] for p in selected_params]
        core1_pool = [p["dynamic_params"]["core1"] for p in selected_params]

        mixed_core0, mixed_core1 = self._mix_cores(core0_pool, core1_pool)
        return {
            "dynamic_params": {
                "core0": mixed_core0,
                "core1": mixed_core1,
            }
        }

    def _mix_cores(
        self,
        core0_pool: List[InstructionProgram],
        core1_pool: List[InstructionProgram],
    ) -> Tuple[InstructionProgram, InstructionProgram]:
        max_cycle = self._get_max_cycle()

        if self.mixer is not None:
            core0 = self.mixer.mix(
                core0_pool,
                max_cycle=max_cycle,
            )
            core1 = self.mixer.mix(
                core1_pool,
                max_cycle=max_cycle,
            )
            return core0, core1

        return deepcopy(core0_pool[0]), deepcopy(core1_pool[0])


@expose
class InterferenceIMGEPExplorer(BaseExplorerFactory):
    config = InterferenceIMGEPConfig
    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, system: System) -> InterferenceIMGEPInstance:
        behavior_map = make_module(
            "behavior_map", system, **self.config.behavior_map_config)
        param_map = make_module("parameter_map", system,
                                **self.config.parameter_map_config)
        mixer = make_module("mixer", **self.config.mixer_config)

        return InterferenceIMGEPInstance(
            parameter_map=param_map,
            behavior_map=behavior_map,
            periode=self.config.periode,
            k=self.config.k,
            mixer=mixer,
        )
