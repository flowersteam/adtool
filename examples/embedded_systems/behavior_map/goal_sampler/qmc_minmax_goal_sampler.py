from typing import List, Optional

import numpy as np
from scipy.stats import qmc

from adtool.examples.embedded_systems.types import GoalSampler


class QMCMinMaxGoalSampler(GoalSampler):
    """Sample goals from history using min/max range expansion and Quasi-Monte Carlo."""

    def __init__(self) -> None:
        self._engine: Optional[qmc.Sobol] = None
        self._feature_size: Optional[int] = None

    def sample(
        self,
        history: List[np.ndarray],
        feature_size: Optional[int],
        min_: Optional[np.ndarray] = None,
        max_: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample goal vector from behavior history using min/max scaling and QMC."""
        if not history:
            if feature_size is None:
                return np.zeros(1, dtype=float)
            return np.zeros(feature_size, dtype=float)

        if min_ is None or max_ is None:
            tab = np.vstack(history)
            min_ = tab.min(axis=0)
            max_ = tab.max(axis=0)

        low = (1 - 0.4 * np.sign(min_)) * min_
        high = 1.4 * max_

        # Degenerate dimensions (high <= low) can happen early in exploration.
        invalid = high <= low
        high[invalid] = low[invalid] + 1e-9

        dims = len(low)
        if self._engine is None or self._feature_size != dims:
            self._engine = qmc.Sobol(d=dims, scramble=True)
            self._feature_size = dims

        # qmc_sample is an array of shape (1, d), so we take [0] to get (d,)
        qmc_sample = self._engine.random(1)[0]

        return low + qmc_sample * (high - low)
