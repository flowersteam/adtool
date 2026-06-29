import copy
import random
from typing import Optional

from examples.embedded_systems.examples.core_interferences.types import InstructionProgram
from examples.embedded_systems.types import ProgramMutator


class RandomInstructionMutator(ProgramMutator):
    """Random add/delete/modify mutator for instruction programs."""

    def __init__(self, seed: Optional[int] = None,
                 num_mutations: int = 5) -> None:
        self.seed = seed
        self.num_mutations = num_mutations

    def mutate(
        self,
        instructions: InstructionProgram,
        *,
        max_cycle: int,
        min_address: int,
        max_address: int,
        num_instructions: int,
    ) -> InstructionProgram:
        """
        Mutate an instruction sequence by adding, deleting, or modifying instructions.

        Args:
            instructions: Original instruction dictionary   
            max_cycle: Maximum cycle number (default: 60)
            max_address: Maximum memory address (default: 19)

        Returns:
            New mutated dictionary
        """
        rng = random.Random(self.seed)

        # Create a deep copy to avoid modifying the original
        mutated = copy.deepcopy(instructions)
        instruction_types = ['read', 'write']

        # Get all possible cycles (0 to max_cycle)
        all_cycles = set(range(0, max_cycle + 1))
        used_cycles = set(mutated.keys())
        available_cycles = list(all_cycles - used_cycles)

        for _ in range(self.num_mutations):
            if len(mutated) > 1:
                mutation_type = rng.choice(['add', 'delete', 'modify'])
            else:
                mutation_type = rng.choice(['add', 'modify'])

            if mutation_type == 'add' and available_cycles:
                # Add a new instruction at an available cycle
                new_cycle = rng.choice(available_cycles)
                instr_type = rng.choice(instruction_types)
                address = rng.randint(min_address, max_address)
                mutated[new_cycle] = (instr_type, address)
                available_cycles.remove(new_cycle)

            elif mutation_type == 'delete' and mutated:
                # Delete a random existing instruction
                cycle_to_delete = rng.choice(list(mutated.keys()))
                del mutated[cycle_to_delete]
                available_cycles.append(cycle_to_delete)

            elif mutation_type == 'modify' and mutated:
                # Modify an existing instruction
                cycle_to_modify = rng.choice(list(mutated.keys()))
                old_type, old_address = mutated[cycle_to_modify]

                # Choose what to modify: type, address, or both
                modify_choice = rng.choice(['type', 'address', 'both'])

                if modify_choice == 'type':
                    # Change instruction type only
                    new_type = 'write' if old_type == 'read' else 'read'
                    mutated[cycle_to_modify] = (new_type, old_address)
                elif modify_choice == 'address':
                    # Change address only
                    new_address = rng.randint(min_address, max_address)
                    mutated[cycle_to_modify] = (old_type, new_address)
                else:
                    # Change both type and address
                    new_type = 'write' if old_type == 'read' else 'read'
                    new_address = rng.randint(min_address, max_address)
                    mutated[cycle_to_modify] = (new_type, new_address)
        if len(mutated) > num_instructions:
            to_del = rng.sample(list(mutated.keys()),
                                len(mutated) - num_instructions)
            for k in to_del:
                del mutated[k]
        return mutated
