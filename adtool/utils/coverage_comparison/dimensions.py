from typing import List

import numpy as np


def align_embeddings(
    embeddings: List[np.ndarray], dim_count: int
) -> List[np.ndarray]:
    aligned: List[np.ndarray] = []
    for emb in embeddings:
        if emb.size != dim_count:
            continue
        aligned.append(emb)
    return aligned
