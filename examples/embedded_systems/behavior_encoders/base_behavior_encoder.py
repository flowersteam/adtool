from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseBehaviorEncoder(ABC):
    """Base class for behavior encoders."""

    @abstractmethod
    def encode(self, raw_output: Dict[str, Any]) -> np.ndarray:
        ...
