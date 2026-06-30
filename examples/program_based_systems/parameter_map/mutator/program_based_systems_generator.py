from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseGenerator(ABC):
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
