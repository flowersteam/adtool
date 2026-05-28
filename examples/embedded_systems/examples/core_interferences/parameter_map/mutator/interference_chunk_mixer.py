import random
from typing import List, Optional, Tuple

from adtool.examples.embedded_systems.examples.core_interferences.helpers.interference_normalization import (
    normalize_instruction_sequences,
)
from adtool.examples.embedded_systems.examples.core_interferences.types import (
    Instruction,
    InstructionProgram,
)
from adtool.examples.embedded_systems.types import ProgramMixer


class ChunkProgramMixer(ProgramMixer):
    """Build a child program from contiguous chunks of nearest-parent programs."""

    def __init__(self, num_parts: int, seed: Optional[int] = None) -> None:
        self.num_parts = max(1, int(num_parts))
        self.seed = seed

    def mix(
        self,
        sequences: List[InstructionProgram],
        *,
        max_cycle: int,
        num_instructions: Optional[int] = None,
    ) -> InstructionProgram:
        rng = random.Random(self.seed)
        parents = normalize_instruction_sequences(
            sequences,
            strict=False,
            context="chunk_mix",
        )
        if not parents or max_cycle < 0:
            return {}

        target_count = self._target_count(parents, num_instructions)
        num_segments = max(self.num_parts, len(parents))
        parent_chunks = [
            self._split_program(program, num_segments)
            for program in parents
        ]
        source_order = self._source_order(len(parents), num_segments, rng)

        mixed_items: List[Tuple[int, Instruction]] = []
        used_cycles = set()
        for segment_idx, parent_idx in enumerate(source_order):
            chunk = parent_chunks[parent_idx][segment_idx]
            if not chunk:
                chunk = self._fallback_chunk(parent_chunks, segment_idx)
            mixed_items.extend(
                self._place_chunk(
                    chunk,
                    segment_idx=segment_idx,
                    num_segments=num_segments,
                    max_cycle=max_cycle,
                    used_cycles=used_cycles,
                )
            )

        mixed_items = sorted(mixed_items, key=lambda item: item[0])
        if target_count is not None and len(mixed_items) > target_count:
            mixed_items = self._evenly_trim(mixed_items, target_count)

        return dict(mixed_items)

    def _target_count(
        self,
        parents: List[InstructionProgram],
        num_instructions: Optional[int],
    ) -> Optional[int]:
        if num_instructions is not None:
            return max(0, int(num_instructions))
        return max((len(parent) for parent in parents), default=0)

    def _split_program(
        self,
        program: InstructionProgram,
        num_segments: int,
    ) -> List[List[Tuple[int, Instruction]]]:
        items = sorted(program.items(), key=lambda item: item[0])
        chunks = []
        for idx in range(num_segments):
            start = round(idx * len(items) / num_segments)
            end = round((idx + 1) * len(items) / num_segments)
            chunks.append(items[start:end])
        return chunks

    def _source_order(
        self,
        num_parents: int,
        num_segments: int,
        rng: random.Random,
    ) -> List[int]:
        source_order = list(range(num_parents))
        while len(source_order) < num_segments:
            source_order.append(rng.randrange(num_parents))
        rng.shuffle(source_order)
        return source_order[:num_segments]

    def _fallback_chunk(
        self,
        parent_chunks: List[List[List[Tuple[int, Instruction]]]],
        segment_idx: int,
    ) -> List[Tuple[int, Instruction]]:
        for chunks in parent_chunks:
            if chunks[segment_idx]:
                return chunks[segment_idx]
        for chunks in parent_chunks:
            for chunk in chunks:
                if chunk:
                    return chunk
        return []

    def _place_chunk(
        self,
        chunk: List[Tuple[int, Instruction]],
        *,
        segment_idx: int,
        num_segments: int,
        max_cycle: int,
        used_cycles: set[int],
    ) -> List[Tuple[int, Instruction]]:
        if not chunk:
            return []

        segment_start = round(segment_idx * (max_cycle + 1) / num_segments)
        segment_end = max(
            segment_start,
            round((segment_idx + 1) * (max_cycle + 1) / num_segments) - 1,
        )
        segment_width = max(1, segment_end - segment_start)
        source_cycles = [cycle for cycle, _ in chunk]
        source_start = min(source_cycles)
        source_span = max(1, max(source_cycles) - source_start)
        scale = min(1.0, segment_width / source_span)

        placed = []
        for old_cycle, instruction in chunk:
            new_cycle = segment_start + round((old_cycle - source_start) * scale)
            new_cycle = min(max(0, new_cycle), max_cycle)
            while new_cycle in used_cycles and new_cycle < max_cycle:
                new_cycle += 1
            while new_cycle in used_cycles and new_cycle > 0:
                new_cycle -= 1
            if new_cycle in used_cycles:
                continue
            used_cycles.add(new_cycle)
            placed.append((new_cycle, instruction))
        return placed

    def _evenly_trim(
        self,
        items: List[Tuple[int, Instruction]],
        target_count: int,
    ) -> List[Tuple[int, Instruction]]:
        if target_count <= 0:
            return []
        if target_count == 1:
            return [items[len(items) // 2]]
        last_idx = len(items) - 1
        selected_indices = {
            round(idx * last_idx / (target_count - 1))
            for idx in range(target_count)
        }
        return [
            item
            for idx, item in enumerate(items)
            if idx in selected_indices
        ]
