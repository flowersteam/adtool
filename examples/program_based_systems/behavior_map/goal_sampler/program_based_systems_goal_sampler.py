from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np


class BaseGoalSampler(ABC):
    """Base class for goal samplers."""

    @abstractmethod
    def sample(
        self,
        history: List[np.ndarray],
        feature_size: Optional[int],
        **kwargs: Any,
    ) -> np.ndarray:
        ...
