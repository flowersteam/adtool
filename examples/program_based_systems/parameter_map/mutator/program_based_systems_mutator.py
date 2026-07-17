from abc import ABC, abstractmethod
from typing import Any

class BaseMutator(ABC):
    """Base class for code mutators."""

    @abstractmethod
    def mutate(
        self,
        instructions: Any,
        *,
        max_cycle: int,
        min_address: int,
        max_address: int,
        num_instructions: int,
    ) -> Any:
        ...
