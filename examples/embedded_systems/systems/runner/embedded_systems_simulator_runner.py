from abc import ABC, abstractmethod
from typing import Any


class BaseSimulatorRunner(ABC):
    """Base class for simulator runners."""

    @abstractmethod
    def run(self, params: Any) -> Any:
        ...
