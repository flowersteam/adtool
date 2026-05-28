import random
from typing import List, Optional

from adtool.examples.embedded_systems.examples.core_interferences.helpers.interference_normalization import (
    normalize_instruction_sequences,
)
from adtool.examples.embedded_systems.examples.core_interferences.types import InstructionProgram
from adtool.examples.embedded_systems.types import ProgramMixer


class PreserveTimeStructureProgramMixer(ProgramMixer):
    """Mixer that keeps relative timing structure within chunks."""

    def __init__(self, num_parts: int, seed: Optional[int] = None) -> None:
        self.num_parts = max(1, int(num_parts))
        self.seed = seed

    def mix(
        self,
        sequences: List[InstructionProgram],
        *,
        max_cycle: int,
    ) -> InstructionProgram:
        """
        Randomly mixes multiple instruction programs into one while preserving
        relative timing delays within each chunk.

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
            context="mix_sequences_preserv",
        )

        # Step 1: sort each program by cycle
        sorted_programs = []
        for program in sequences:
            instrs = sorted(program.items(), key=lambda x: x[0])
            sorted_programs.append(instrs)

        # Step 2: split each program into num_parts chunks
        chunks = []
        for instrs in sorted_programs:
            if not instrs:
                continue

            chunk_size = max(1, len(instrs) // self.num_parts)
            for i in range(0, len(instrs), chunk_size):
                chunk = instrs[i:i + chunk_size]
                chunks.append(chunk)

        # Step 3: shuffle chunks
        rng.shuffle(chunks)

        # Step 4: flatten chunks into a single instruction list
        mixed_instrs = []
        for chunk in chunks:
            mixed_instrs.extend(chunk)

        if not mixed_instrs:
            return {}

        # Step 5: assign new cycles while preserving relative timing within chunks
        mixed_program = {}
        current_cycle = 1

        # Track where each chunk starts and ends in the flattened list
        chunk_boundaries = []
        start_idx = 0
        for chunk in chunks:
            chunk_boundaries.append((start_idx, start_idx + len(chunk)))
            start_idx += len(chunk)

        # Process each chunk separately to preserve internal timing
        for chunk_start, chunk_end in chunk_boundaries:
            chunk_instrs = mixed_instrs[chunk_start:chunk_end]

            if not chunk_instrs:
                continue

            # Get original cycles from the chunk
            original_cycles = [cycle for cycle, _ in chunk_instrs]

            # Calculate relative offsets from the first instruction in the chunk
            base_cycle = original_cycles[0]
            offsets = [cycle - base_cycle for cycle in original_cycles]

            # Ensure we have space for the entire chunk
            max_offset = max(offsets) if offsets else 0
            if current_cycle + max_offset > max_cycle:
                # Not enough space, need to pack tighter or handle differently
                # For now, we'll pack them with minimum spacing of 1
                new_base = current_cycle
            else:
                new_base = current_cycle

            # Assign new cycles based on preserved offsets
            for idx, (_, instr) in enumerate(chunk_instrs):
                new_cycle = new_base + offsets[idx]
                mixed_program[new_cycle] = instr

            # Update current_cycle to after this chunk
            current_cycle = max(current_cycle, new_base + max_offset + 1)

            # If we exceed max_cycle, we need to compress the remaining chunks
            if current_cycle > max_cycle:
                # Fallback: pack remaining instructions sequentially with no gaps
                remaining_instrs = mixed_instrs[chunk_end:]
                for idx, (_, instr) in enumerate(remaining_instrs):
                    mixed_program[current_cycle + idx] = instr
                break

        # Optional: compress the entire program if cycles exceed max_cycle
        if mixed_program and max(mixed_program.keys()) > max_cycle:
            # Recompress to fit within max_cycle
            compressed = {}
            new_cycle = 1
            sorted_cycles = sorted(mixed_program.keys())
            for old_cycle in sorted_cycles:
                compressed[new_cycle] = mixed_program[old_cycle]
                new_cycle += 1
            mixed_program = compressed

        return mixed_program
