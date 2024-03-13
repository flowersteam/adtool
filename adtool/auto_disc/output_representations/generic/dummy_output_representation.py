import torch
from adtool.auto_disc.output_representations import BaseOutputRepresentation
from adtool.auto_disc.utils.spaces.utils import distance


class DummyOutputRepresentation(BaseOutputRepresentation):
    """
    Empty OutputRepresentation used when no representation of the system's output mut be used.
    """

    def __init__(self, wrapped_input_space_key=None, **kwargs) -> None:
        super().__init__(wrapped_input_space_key=wrapped_input_space_key, **kwargs)

    def map(self, input, is_output_new_discovery, **kwargs):
        return input

    def calc_distance(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # L2 + add regularizer to avoid dead outcomes
        dist = distance.calc_l2(embedding_a, embedding_b)
        return dist
