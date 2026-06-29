import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from examples.embedded_systems.parameter_map.embedded_systems_parameter_map import (
    BaseParameterMap,
)
from examples.embedded_systems.examples.core_interferences.helpers.interference_normalization import (
    normalize_instruction_program,
)
from examples.embedded_systems.examples.core_interferences.systems.InterferenceSystem import (
    InterferenceSystem,
)
from examples.embedded_systems.examples.core_interferences.types import (
    InstructionProgram,
    InterferenceParamsPayload,
)
from examples.embedded_systems.helpers.module_factory import make_module


@dataclass
class InterferenceParams:
    num_instructions: int = 10
    min_address_core0: int = 0
    max_address_core0: int = 20
    min_address_core1: int = 21
    max_address_core1: int = 40
    max_cycle: int = 400


class InterferenceParameterMap(BaseParameterMap):
    """Parameter map RANDOM generation and mutation policy."""

    def __init__(
            self,
            system: InterferenceSystem,
            premap_key: str = "params",
            param_obj: InterferenceParams = None,
            generator_config: Optional[Dict[str, Any]] = {
                "path": "examples.embedded_systems.examples.core_interferences.parameter_map.mutator.interference_random_instruction_generator.RandomInstructionGenerator"
            },
            mutator_config: Optional[Dict[str, Any]] = {
                "path": "examples.embedded_systems.examples.core_interferences.parameter_map.mutator.interference_random_instruction_mutator.RandomInstructionMutator",
                "num_mutations": 2,
            },
            mixer_config: Optional[Dict[str, Any]] = None,
            **config_decorator_kwargs,
    ) -> None:
        _ = system
        super().__init__(premap_key=premap_key)

        if param_obj is None:
            param_obj = InterferenceParams()

        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(
                param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key
        self.param_obj = param_obj
        self.generator = make_module("generator", **generator_config)
        self.mutator = make_module("mutator", **mutator_config)
        self.mixer = (
            make_module("mixer", **mixer_config)
            if mixer_config is not None
            else None
        )

    def sample(self) -> InterferenceParamsPayload:
        # RANDOM exploration: generate two independent programs,
        # one per core, within their respective address domains.
        p = self.param_obj
        core0: InstructionProgram = self.generator.generate(
            num_instructions=p.num_instructions,
            min_address=p.min_address_core0,
            max_address=p.max_address_core0,
            max_cycle=p.max_cycle,
        )
        core1: InstructionProgram = self.generator.generate(
            num_instructions=p.num_instructions,
            min_address=p.min_address_core1,
            max_address=p.max_address_core1,
            max_cycle=p.max_cycle,
        )

        return {
            "dynamic_params": {
                "core0": core0,
                "core1": core1,
            }
        }
        # Returned shape is intentionally nested to match System.map expectations:
        # input["params"]["dynamic_params"]["core0"|"core1"].

    def mutate(
        self,
        parameter_dict: Union[
            InterferenceParamsPayload,
            List[InterferenceParamsPayload],
        ],
    ) -> InterferenceParamsPayload:
        intermed = self._prepare_parent_payload(parameter_dict)

        p = self.param_obj
        dyn = intermed["dynamic_params"]
        core0 = normalize_instruction_program(
            dyn["core0"],
            strict=True,
            context="InterferenceParameterMap.core0",
        )
        core1 = normalize_instruction_program(
            dyn["core1"],
            strict=True,
            context="InterferenceParameterMap.core1",
        )

        # Mutation is intentionally asymmetric by address range: each core keeps
        # its own address constraints to preserve interference structure.
        dyn["core0"] = self.mutator.mutate(
            instructions=core0,
            max_cycle=p.max_cycle,
            min_address=p.min_address_core0,
            max_address=p.max_address_core0,
            num_instructions=p.num_instructions,
        )
        dyn["core1"] = self.mutator.mutate(
            instructions=core1,
            max_cycle=p.max_cycle,
            min_address=p.min_address_core1,
            max_address=p.max_address_core1,
            num_instructions=p.num_instructions,
        )

        return intermed

    def _prepare_parent_payload(
        self,
        parameter_dict: Union[
            InterferenceParamsPayload,
            List[InterferenceParamsPayload],
        ],
    ) -> InterferenceParamsPayload:
        if isinstance(parameter_dict, list):
            if not parameter_dict:
                return self.sample()
            if len(parameter_dict) == 1 or self.mixer is None:
                return deepcopy(parameter_dict[0])
            return self._mix_parent_payloads(parameter_dict)

        return deepcopy(parameter_dict)

    def _mix_parent_payloads(
        self,
        parents: List[InterferenceParamsPayload],
    ) -> InterferenceParamsPayload:
        p = self.param_obj
        core0_pool: List[InstructionProgram] = []
        core1_pool: List[InstructionProgram] = []

        for index, parent in enumerate(parents):
            dyn = parent["dynamic_params"]
            core0_pool.append(
                normalize_instruction_program(
                    dyn["core0"],
                    strict=True,
                    context=f"InterferenceParameterMap.mix_parent{index}.core0",
                )
            )
            core1_pool.append(
                normalize_instruction_program(
                    dyn["core1"],
                    strict=True,
                    context=f"InterferenceParameterMap.mix_parent{index}.core1",
                )
            )

        return {
            "dynamic_params": {
                "core0": self.mixer.mix(
                    core0_pool,
                    max_cycle=p.max_cycle,
                    num_instructions=p.num_instructions,
                ),
                "core1": self.mixer.mix(
                    core1_pool,
                    max_cycle=p.max_cycle,
                    num_instructions=p.num_instructions,
                ),
            }
        }

    def map(self, input: Dict[str, Any], override_existing: bool = True) -> Dict[str, Any]:
        intermed = deepcopy(input)
        # Inject a fresh sample only when requested or when params are missing.
        if (override_existing and self.postmap_key in intermed) or (
                self.postmap_key not in intermed
        ):
            intermed[self.postmap_key] = self.sample()
        # If override is disabled and params already exist, they pass through
        # untouched so explorer-selected candidates are preserved.
        return intermed
