from abc import ABC
from typing import Any, Dict


class BaseSimulator(ABC):
    """Base class for simulator backends."""

    def __init__(self, **config: Any) -> None:
        self.config: Dict[str, Any] = dict(config)
