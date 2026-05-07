from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCodeOperator(ABC):
    """Shared base for code-related operators (generation, mutation, mixing)."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    def normalize_code(self, code: Any) -> Any:
        return code

    def validate_code(self, code: Any) -> bool:
        return True


class BaseGenerator(BaseCodeOperator, ABC):
    """Base class for code generators."""

    @abstractmethod
    def generate(
        self,
        *,
        num_instructions: int,
        max_cycle: int,
        min_address: int,
        max_address: int,
    ) -> Any:
        ...
