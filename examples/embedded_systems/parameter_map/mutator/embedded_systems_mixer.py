from abc import ABC, abstractmethod
from typing import Any, List

class BaseMixer(ABC):
    """Base class for code mixers."""

    @abstractmethod
    def mix(
        self,
        sequences: List[Any],
        *,
        max_cycle: int,
    ) -> Any:
        ...
