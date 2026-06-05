from abc import ABC
from typing import Any, Dict

# The simulator can be anything, and can be quite large.
# The objective of this base class is to provide a common interface between all simulators in order
# to be able to use them interchangeably in the rest of the codebase, and to be able to easily swap them out.

class BaseSimulator(ABC):
    """Base class for simulator backends."""

    def __init__(self, **config: Any) -> None:
        self.config: Dict[str, Any] = dict(config)
