from typing_extensions import Dict, Literal, Tuple, TypedDict

from adtool.utils.factory import ObjectSpec


InstructionType = Literal["read", "write"]
Instruction = Tuple[InstructionType, int]
InstructionProgram = Dict[int, Instruction]


class InterferenceDynamicParams(TypedDict):
    core0: InstructionProgram
    core1: InstructionProgram


class InterferenceParamsPayload(TypedDict):
    dynamic_params: InterferenceDynamicParams


InterferenceSimulatorConfig = ObjectSpec
InterferenceSimulatorRunnerConfig = ObjectSpec
