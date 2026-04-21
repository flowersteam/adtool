import random

def generate_instruction_sequence(num_instructions=None, max_cycle=60, min_address=0,max_address=19):
    """
    Generate a random dictionary of assembly instructions.
    
    Args:
        num_instructions: Number of instructions to generate (if None, random between 1-20)
        max_cycle: Maximum cycle number (default: 60)
        max_address: Maximum memory address (default: 19)
    
    Returns:
        Dictionary with format {cycle: (type, address)}
    """
    if num_instructions is None:
        num_instructions = random.randint(1, 20)  # Random number of instructions
        # Ensure we don't generate more instructions than available cycles
    num_instructions = min(num_instructions, max_cycle + 1)
    
    instructions = {}
    instruction_types = ['read', 'write']
    
    # Generate unique cycle numbers
    cycles = random.sample(range(0, max_cycle + 1), num_instructions)
    
    for cycle in cycles:
        instr_type = random.choice(instruction_types)
        address = random.randint(min_address, max_address)
        instructions[cycle] = (instr_type, address)
    
    return dict(sorted(instructions.items()))
