import heapq
import random
from typing import List, Optional

from examples.embedded_systems.examples.core_interferences.helpers.interference_normalization import (
    normalize_instruction_sequences,
)
from examples.embedded_systems.examples.core_interferences.types import InstructionProgram
from examples.embedded_systems.types import ProgramMixer


class InterleavingProgramMixer(ProgramMixer):
    """Mixer that interleaves instructions from multiple parents."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    def mix(
        self,
        sequences: List[InstructionProgram],
        *,
        max_cycle: int,
    ) -> InstructionProgram:
        """
        Randomly mixes multiple instruction programs by interleaving instructions
        while preserving relative timing delays within each original program.

        Args:
            sequences (list[dict]): List of programs {cycle: (type, address)}
            seed (int | None): Random seed
            max_cycle (int): Maximum cycle number in output

        Returns:
            dict: Mixed program {cycle: (type, address)}
        """

        rng = random.Random(self.seed)

        # Step 0: normalize payloads and drop malformed programs with warnings.
        sequences = normalize_instruction_sequences(
            sequences,
            strict=False,
            context="mix_sequences_interleaved",
        )

        # Step 1: sort each program by cycle and convert to list of instructions
        # Also store the original timing offsets
        programs = []
        for program in sequences:
            if not program:
                continue

            # Sort by cycle and get list of (cycle, instruction)
            instrs = sorted(program.items(), key=lambda x: x[0])

            # Calculate offsets from the first instruction
            base_cycle = instrs[0][0]
            timed_instrs = [(cycle - base_cycle, instr)
                            for cycle, instr in instrs]

            programs.append(timed_instrs)

        if not programs:
            return {}

        # Step 2: create a priority queue for random selection
        # Each entry: (random_key, program_index, instruction_index)
        # This allows random selection while maintaining program order
        heap = []

        for prog_idx, program in enumerate(programs):
            if program:
                # Generate random priority for the first instruction of each program
                random_priority = rng.random()
                heapq.heappush(heap, (random_priority, prog_idx, 0))

        # Step 3: interleave instructions while preserving timing offsets
        mixed_program = {}
        current_cycle = 1

        # Keep track of the base cycle for each program as we place instructions
        program_bases = [None] * len(programs)

        while heap:
            # Get the next instruction to place
            _, prog_idx, instr_idx = heapq.heappop(heap)

            program = programs[prog_idx]
            offset, instr = program[instr_idx]

            # Determine the cycle for this instruction
            if program_bases[prog_idx] is None:
                # First instruction from this program
                program_bases[prog_idx] = current_cycle
                new_cycle = current_cycle
            else:
                # Calculate new cycle based on preserved offset from program base
                new_cycle = program_bases[prog_idx] + offset

            # Check if we exceed max_cycle
            if new_cycle > max_cycle:
                # Need to shift the entire program or handle overflow
                # Option: shift all remaining instructions from this program to fit
                shift = max_cycle - new_cycle
                if shift < 0:
                    # Find minimum cycle we can place this at
                    min_possible = max(program_bases[prog_idx], current_cycle)
                    new_cycle = min_possible + offset

                    # If still too large, we need to reposition the whole program
                    if new_cycle > max_cycle:
                        # Find a spot where the entire remaining program fits
                        remaining_instructions = len(program) - instr_idx
                        max_offset = program[-1][0]

                        # Try to place at current_cycle
                        if current_cycle + max_offset <= max_cycle:
                            program_bases[prog_idx] = current_cycle - offset
                            new_cycle = current_cycle
                        else:
                            # Cannot fit remaining instructions, skip this program
                            continue

            # Place the instruction
            if new_cycle in mixed_program:
                # Cycle collision - shift forward until we find an empty cycle
                while new_cycle in mixed_program and new_cycle <= max_cycle:
                    new_cycle += 1
                if new_cycle > max_cycle:
                    continue

            mixed_program[new_cycle] = instr
            current_cycle = max(current_cycle, new_cycle + 1)

            # Schedule the next instruction from this program if available
            if instr_idx + 1 < len(program):
                random_priority = rng.random()
                heapq.heappush(
                    heap, (random_priority, prog_idx, instr_idx + 1))

        # Step 4: compress if necessary to fit within max_cycle
        if mixed_program and max(mixed_program.keys()) > max_cycle:
            compressed = {}
            new_cycle = 1
            sorted_cycles = sorted(mixed_program.keys())
            for old_cycle in sorted_cycles:
                compressed[new_cycle] = mixed_program[old_cycle]
                new_cycle += 1
            mixed_program = compressed

        return mixed_program
