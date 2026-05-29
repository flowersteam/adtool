from typing_extensions import Any, Dict, Literal, Tuple, TypedDict


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
