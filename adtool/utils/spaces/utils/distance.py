import torch

"""
    all distance function
"""


def calc_l2(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Euclidean distance between 2 vectors

    Ars:
        embedding_a: first vector
        embedding_b: second vector
    Returns:
        dist: The distance between the 2 vectors
    """
    # L2 + add regularizer to avoid dead outcomes
    dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt() - 0.5 * embedding_b.pow(
        2
    ).sum(-1).sqrt()
    return dist
