import random
from typing import Optional

from examples.core_interference.types import InstructionProgram, ProgramGenerator


class RandomInstructionGenerator(ProgramGenerator):
    """Random instruction program generator."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    def generate(
        self,
        *,
        num_instructions: int,
        max_cycle: int,
        min_address: int,
        max_address: int,
    ) -> InstructionProgram:
        """
        Generate a random dictionary of assembly instructions.

        Args:
            num_instructions: Number of instructions to generate (if None, random between 1-20)
            max_cycle: Maximum cycle number (default: 60)
            max_address: Maximum memory address (default: 19)

        Returns:
            Dictionary with format {cycle: (type, address)}
        """
        rng = random.Random(self.seed)

        bounded_num_instructions = min(max(0, num_instructions), max_cycle + 1)
        if bounded_num_instructions == 0:
            return {}

        instructions: InstructionProgram = {}
        instruction_types = ["read", "write"]
        cycles = rng.sample(range(0, max_cycle + 1), bounded_num_instructions)

        for cycle in cycles:
            instr_type = rng.choice(instruction_types)
            address = rng.randint(min_address, max_address)
            instructions[cycle] = (instr_type, address)

        return dict(sorted(instructions.items()))
