import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

from adtool.utils.leaf.Leaf import Leaf
from examples.core_interference.helpers.codegeneration import (
	generate_instruction_sequence,
)
from examples.core_interference.helpers.modifiers.mutation import (
	mutate_instruction_sequence,
)
from examples.core_interference.helpers.modifiers.normalization import (
	normalize_instruction_program,
)
from examples.core_interference.systems.InterferenceSystem import InterferenceSystem
from examples.core_interference.types import (
	InstructionProgram,
	InterferenceParamsPayload,
)


@dataclass
class InterferenceParams:
	num_mutations: int = 5
	num_instructions: int = 10
	min_address_core0: int = 0
	max_address_core0: int = 20
	min_address_core1: int = 21
	max_address_core1: int = 40
	max_cycle: int = 400


class InterferenceParameterMap(Leaf):
	"""Parameter map RANDOM generation and mutation policy."""

	def __init__(
		self,
		system: InterferenceSystem,
		premap_key: str = "params",
		param_obj: InterferenceParams = None,
		**config_decorator_kwargs,
	) -> None:
		super().__init__()
		_ = system

		if param_obj is None:
			param_obj = InterferenceParams()

		if len(config_decorator_kwargs) > 0:
			param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

		self.premap_key = premap_key
		self.param_obj = param_obj

	def sample(self) -> InterferenceParamsPayload:
		# RANDOM exploration: generate two independent programs,
		# one per core, within their respective address domains.
		p = self.param_obj
		core0: InstructionProgram = generate_instruction_sequence(
			num_instructions=p.num_instructions,
			min_address=p.min_address_core0,
			max_address=p.max_address_core0,
			max_cycle=p.max_cycle,
		)
		core1: InstructionProgram = generate_instruction_sequence(
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

	def mutate(self, parameter_dict: InterferenceParamsPayload) -> InterferenceParamsPayload:
		intermed = deepcopy(parameter_dict)

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
		dyn["core0"] = mutate_instruction_sequence(
			instructions=core0,
			num_mutations=p.num_mutations,
			max_cycle=p.max_cycle,
			min_address=p.min_address_core0,
			max_address=p.max_address_core0,
			num_instructions=p.num_instructions,
		)
		dyn["core1"] = mutate_instruction_sequence(
			instructions=core1,
			num_mutations=p.num_mutations,
			max_cycle=p.max_cycle,
			min_address=p.min_address_core1,
			max_address=p.max_address_core1,
			num_instructions=p.num_instructions,
		)

		return intermed

	def map(self, input: Dict[str, Any], override_existing: bool = True) -> Dict[str, Any]:
		intermed = deepcopy(input)
		# Inject a fresh sample only when requested or when params are missing.
		if (override_existing and self.premap_key in intermed) or (
			self.premap_key not in intermed
		):
			intermed[self.premap_key] = self.sample()
		# If override is disabled and params already exist, they pass through
		# untouched so explorer-selected candidates are preserved.
		return intermed
