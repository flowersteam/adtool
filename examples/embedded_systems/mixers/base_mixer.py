from abc import ABC, abstractmethod
from typing import Any, List

from examples.embedded_systems.generators.base_generator import BaseCodeOperator


class BaseMixer(BaseCodeOperator, ABC):
    """Base class for code mixers."""

    @abstractmethod
    def mix(
        self,
        sequences: List[Any],
        *,
        max_cycle: int,
    ) -> Any:
        ...
