from abc import ABC, abstractmethod
from typing import Any, List

from examples.embedded_systems.generators.embedded_systems_generator import BaseCodeOperator


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
