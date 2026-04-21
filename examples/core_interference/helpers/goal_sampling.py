from typing import List, Optional

import numpy as np
from scipy.stats import qmc
from scipy.spatial import distance

_LOW_SCALE = 0.4
_HIGH_SCALE = 1.4
_CANDIDATE_BATCH_SIZE = 64
_K_NEIGHBORS = 8
_EPS = 1e-9
_SOBOL_SEED = 0
_SOBOL_ENGINES = {}


def _get_sobol_engine(dimension: int) -> qmc.Sobol:
    engine = _SOBOL_ENGINES.get(dimension)
    if engine is None:
        engine = qmc.Sobol(d=dimension, scramble=True, seed=_SOBOL_SEED)
        _SOBOL_ENGINES[dimension] = engine
    return engine


def _sample_sobol_candidates(low: np.ndarray, high: np.ndarray, n: int) -> np.ndarray:
    engine = _get_sobol_engine(low.size)
    unit = engine.random(n)
    return low + unit * (high - low)


def _select_most_novel_candidate(
    candidates: np.ndarray,
    archive: np.ndarray,
) -> np.ndarray:
    if archive.shape[0] == 0:
        return candidates[0]

    k_eff = min(_K_NEIGHBORS, archive.shape[0])
    
    distances_sq = distance.cdist(candidates, archive, metric='sqeuclidean')
    
    nearest_sq = np.partition(distances_sq, kth=k_eff - 1, axis=1)[:, :k_eff]
    novelty_scores = nearest_sq.mean(axis=1)
    
    return candidates[int(np.argmax(novelty_scores))]

def sample_goal_from_history(
    history: List[np.ndarray],
    feature_size: Optional[int],
) -> np.ndarray:
    """Sample goals with Sobol-RQMC candidates + kNN novelty selection."""
    if not history:
        if feature_size is None:
            return np.zeros(1, dtype=float)
        return np.zeros(feature_size, dtype=float)
    
    # Boundary expansion
    tab = np.vstack(history)
    min_ = tab.min(axis=0)
    max_ = tab.max(axis=0)
    ptp = max_ - min_  # The range of each dimension

    # If range is exactly 0 (e.g., all goals are identical so far), give it a default spread
    ptp = np.where(ptp == 0, 1.0, ptp)

    low = min_ - _LOW_SCALE * ptp
    high = max_ + _HIGH_SCALE * ptp

    # Degenerate dimensions (high <= low) can happen early in exploration.
    invalid = high <= low
    high[invalid] = low[invalid] + _EPS

    candidates = _sample_sobol_candidates(low, high, _CANDIDATE_BATCH_SIZE)
    return _select_most_novel_candidate(candidates, tab)

