import random
import warnings
from typing import List, Optional

from examples.core_interference.helpers.normalization import (
    normalize_instruction_sequences,
)
from examples.core_interference.types import InstructionProgram


class ChunkProgramMixer:
    """Chunk-based mixer using random chunk shuffling."""

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
        Randomly mixes multiple instruction programs into one.

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
            context="mix_sequences",
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

        if max_cycle < 1:
            warnings.warn(
                f"mix_sequences: invalid max_cycle={max_cycle}; returning empty program",
                RuntimeWarning,
            )
            return {}

        # Step 5: assign new increasing random cycles
        num_instrs = len(mixed_instrs)
        if num_instrs > max_cycle:
            warnings.warn(
                "mix_sequences: instruction count exceeds available cycle slots; "
                f"truncating from {num_instrs} to {max_cycle}",
                RuntimeWarning,
            )
            mixed_instrs = mixed_instrs[:max_cycle]
            num_instrs = len(mixed_instrs)

        available_cycles = sorted(
            rng.sample(range(1, max_cycle + 1), k=num_instrs)
        )

        # Step 6: build final program
        mixed_program = {
            cycle: instr
            for cycle, (_, instr) in zip(available_cycles, mixed_instrs)
        }

        return mixed_program
