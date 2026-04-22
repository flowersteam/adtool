from typing import Any, Dict, List, Literal, Protocol, Tuple, TypedDict

import numpy as np


InstructionType = Literal["read", "write"]
Instruction = Tuple[InstructionType, int]
InstructionProgram = Dict[int, Instruction]


class InterferenceDynamicParams(TypedDict):
    core0: InstructionProgram
    core1: InstructionProgram


class InterferenceParamsPayload(TypedDict):
    dynamic_params: InterferenceDynamicParams


class GoalSampler(Protocol):
    def sample(self, history: List[np.ndarray], feature_size: int | None) -> np.ndarray:
        ...


class ProgramMixer(Protocol):
    def mix(
        self,
        sequences: List[InstructionProgram],
        *,
        max_cycle: int,
        num_parts: int,
    ) -> InstructionProgram:
        ...


class ProgramMutator(Protocol):
    def mutate(self, program: InstructionProgram) -> InstructionProgram:
        ...


class BehaviorEncoder(Protocol):
    def encode(self, raw_output: Dict[str, Any]) -> np.ndarray:
        ...


class SimulatorRunner(Protocol):
    def run(self, params: InterferenceDynamicParams) -> Dict[str, Any]:
        ...