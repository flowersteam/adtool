from pydantic import BaseModel, ConfigDict
from typing_extensions import Dict, Literal, Tuple, TypedDict


InstructionType = Literal["read", "write"]
Instruction = Tuple[InstructionType, int]
InstructionProgram = Dict[int, Instruction]


class InterferenceDynamicParams(TypedDict):
    core0: InstructionProgram
    core1: InstructionProgram


class InterferenceParamsPayload(TypedDict):
    dynamic_params: InterferenceDynamicParams


class InterferenceSimulatorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    cycles: int
    num_banks: int
    num_addr: int


class InterferenceSimulatorRunnerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
