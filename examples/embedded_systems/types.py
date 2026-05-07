from typing_extensions import Any, Dict, List, Literal, Protocol, Tuple, TypedDict

import numpy as np


InstructionType = Literal["read", "write"]
Instruction = Tuple[InstructionType, int]
InstructionProgram = Dict[int, Instruction]


class InterferenceDynamicParams(TypedDict):
    core0: InstructionProgram
    core1: InstructionProgram


class InterferenceParamsPayload(TypedDict):
    dynamic_params: InterferenceDynamicParams


class InterferenceSimulatorConfig(TypedDict):
    path: str
    cycles: int
    num_banks: int
    num_addr: int


class InterferenceSimulatorRunnerConfig(TypedDict):
    path: str


class GoalSampler(Protocol):
    def sample(self, history: List[np.ndarray], feature_size: int | None) -> np.ndarray:
        ...


class ProgramMixer(Protocol):
    def mix(
        self,
        sequences: List[InstructionProgram],
        *,
        max_cycle: int,
    ) -> InstructionProgram:
        ...


class ProgramMutator(Protocol):
    def mutate(
        self,
        instructions: InstructionProgram,
        *,
        max_cycle: int,
        min_address: int,
        max_address: int,
        num_instructions: int,
    ) -> InstructionProgram:
        ...


class ProgramGenerator(Protocol):
    def generate(
        self,
        *,
        num_instructions: int,
        max_cycle: int,
        min_address: int,
        max_address: int,
    ) -> InstructionProgram:
        ...


class BehaviorEncoder(Protocol):
    def encode(self, raw_output: Dict[str, Any]) -> np.ndarray:
        ...


class Simulator(Protocol):
    Any

class SimulatorRunner(Protocol):
    def run(self, params: Any) -> Any:
        ...