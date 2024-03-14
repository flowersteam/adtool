import numbers
from typing import Any, Dict

import torch
from adtool.utils.spaces import BaseSpace


class MultiBinarySpace(BaseSpace):
    """
    An n-shape binary space.

    The argument to MultiBinarySpace defines n, which could be a number or a `list` of numbers.

    Example Usage:

    >> self.genome_space = spaces.MultiBinarySpace(5)

    >> self.genome_space.sample()

        array([0,1,0,1,0], dtype =int8)

    >> self.genome_space = spaces.MultiBinarySpace([3,2])

    >> self.genome_space.sample()

        array([[0, 0],
               [0, 1],
               [1, 1]], dtype=int8)

    """

    def __init__(self, n: int, indpb: float = 1.0) -> None:
        """
        Init the elements useful to the spaces

        Args:
            n: number of element
            indpb: Independent probability for each attribute to be exchanged
        """
        self._indpb = indpb

        super(MultiBinarySpace, self).__init__((n,), torch.int8)

    def initialize(self, parent_obj: object) -> None:
        """
        Initialize the space.

        Args:
            parent_obj: The current autodisc module
        """
        # Apply potential binding
        super().initialize(parent_obj)

        if isinstance(self._indpb, numbers.Number):
            self._indpb = torch.full(self.shape, self._indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(self._indpb, dtype=torch.float64)

    def sample(self) -> torch.Tensor:
        """
        Generates a random sample.

        Return:
            The return value is torch of the sample value
        """
        return torch.randint(low=0, high=2, size=self.shape, dtype=self.dtype)

    def mutate(self, x: Any) -> torch.Tensor:
        """
        Apply a mutation on x

        Args:
            x: The variable to mutate
        Returns:
            x: the result of mutation
        """
        # TODO implement mutator?
        mutate_mask = torch.rand(self.shape) < self.indpb
        x = torch.where(mutate_mask, (~x.bool()).type(self.dtype), x)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x: Any) -> bool:
        """
        check if x is a tensor of 0 and 1 of a given size

        Args:
            x: The value to check
        Returns:
            The return value is True if x is a tensor of 0 and 1 of a given size False otherwise
        """
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.tensor(x)  # Promote list to array for contains check
        if self.shape != x.shape:
            return False
        return ((x == 0) | (x == 1)).all()

    def clamp(self, x: Any) -> torch.Tensor:
        # TODO?
        return x

    def __repr__(self):
        """
        Give a string representation of the class's object

        Returns:
            The return value is a string which represent our object
        """
        return "MultiBinarySpace({})".format(self.shape[0])

    def __eq__(self, other: object) -> bool:
        """
        Check if the other object are equal to the current object

        Args:
            other: An object
        Returns:
            The return value is True if self and other are equal False otherwise
        """
        return isinstance(other, MultiBinarySpace) and self.shape[0] == other.shape[0]

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the object into JSON

        Returns:
            The JSON of the object
        """
        dict = super().to_json()
        dict["indpb"] = self._indpb
        return dict
