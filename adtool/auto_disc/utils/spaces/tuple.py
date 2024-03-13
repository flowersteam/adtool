from typing import Tuple

import torch
from adtool.auto_disc.utils.spaces import BaseSpace


class TupleSpace(BaseSpace):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """

    def __init__(self, spaces):
        """
        Init the elements the spaces

        Args:
            spaces: elements spaces
        """
        self.spaces = spaces
        for space in spaces:
            assert isinstance(
                space, BaseSpace
            ), "Elements of the tuple must be instances of leniasearch.Space"
        super(TupleSpace, self).__init__(None, None)

    def initialize(self, parent_obj):
        """
        Initialize the space.

        Args:
            parent_obj: The current autodisc module
        """
        super().initialize(parent_obj)

    def sample(self) -> Tuple[torch.Tensor]:
        """
        Generates samples for each spaces.

        Return:
            The return value is tuple of torch of samples values
        """
        return tuple([space.sample() for space in self.spaces])

    def mutate(self, x) -> Tuple:
        """
        Apply a mutation on each spaces

        Args:
            x: The variable to mutate
        Returns:
            x: the result of mutation
        """
        return tuple([space.mutate(part) for (space, part) in zip(self.spaces, x)])

    def contains(self, x) -> bool:
        """
        Check if x included in spaces

        Args:
            x: The value to check
        Returns:
            The return value is True if x is included False otherwise
        """
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return (
            isinstance(x, tuple)
            and len(x) == len(self.spaces)
            and all(space.contains(part) for (space, part) in zip(self.spaces, x))
        )

    def clamp(self, x) -> Tuple:
        """
        Set each element of tuple to an accpetable value of his space

        Args:
            x: tuple elemnts
        Returns:
            The return value is clamped value of each element
        """
        return tuple([space.clamp(x) for space in self.spaces])

    def __repr__(self):
        """
        Give a string representation of the class's object

        Returns:
            The return value is a string which represent our object
        """
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def __getitem__(self, index: int) -> BaseSpace:
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TupleSpace) and self.spaces == other.spaces
