from typing import List, Optional

import numpy as np


class RandomMinMaxGoalSampler:
    """Sample goals from history using min/max range expansion."""

    def sample(
        self,
        history: List[np.ndarray],
        feature_size: Optional[int],
    ) -> np.ndarray:
        """Sample goal vector from behavior history using min/max scaling."""
        if not history:
            if feature_size is None:
                return np.zeros(1, dtype=float)
            return np.zeros(feature_size, dtype=float)

        tab = np.vstack(history)
        min_ = tab.min(axis=0)
        max_ = tab.max(axis=0)

        low = (1 - 0.4 * np.sign(min_)) * min_
        high = 1.4 * max_

        # Degenerate dimensions (high <= low) can happen early in exploration.
        invalid = high <= low
        high[invalid] = low[invalid] + 1e-9

        return np.random.uniform(low, high)
