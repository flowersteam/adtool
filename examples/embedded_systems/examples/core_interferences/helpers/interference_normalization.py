from typing import Any, Dict, List, Tuple
import warnings


def normalize_instruction_program(
    program: Any,
    *,
    strict: bool = True,
    context: str = "program",
) -> Dict[int, Tuple[str, int]]:
    """Normalize instruction payload into {int_cycle: (op_type, int_addr)}."""
    def fail(message: str) -> None:
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning)

    if not isinstance(program, dict):
        fail(f"{context}: expected dict payload, got {type(program).__name__}")
        return {}

    normalized: Dict[int, Tuple[str, int]] = {}
    for cycle_raw, operation_raw in program.items():
        try:
            if isinstance(cycle_raw, bool):
                raise ValueError("boolean")
            cycle = int(cycle_raw)
        except (TypeError, ValueError):
            fail(f"{context}: invalid cycle key {cycle_raw!r}")
            continue

        if not isinstance(operation_raw, (list, tuple)) or len(operation_raw) != 2:
            fail(
                f"{context}: invalid operation at cycle {cycle!r}; "
                "expected (type, address)"
            )
            continue

        try:
            address = int(operation_raw[1])
        except (TypeError, ValueError):
            fail(
                f"{context}: invalid address {operation_raw[1]!r} "
                f"at cycle {cycle!r}"
            )
            continue

        normalized[cycle] = (str(operation_raw[0]), address)

    return dict(sorted(normalized.items()))


def normalize_instruction_sequences(
    sequences: List[Any],
    *,
    strict: bool = False,
    context: str = "mix",
) -> List[Dict[int, Tuple[str, int]]]:
    """Normalize list of programs, keep non-empty ones."""
    normalized_programs: List[Dict[int, Tuple[str, int]]] = []
    for idx, program in enumerate(sequences):
        normalized = normalize_instruction_program(
            program, strict=strict, context=f"{context}[{idx}]"
        )
        if normalized:
            normalized_programs.append(normalized)
    return normalized_programs
