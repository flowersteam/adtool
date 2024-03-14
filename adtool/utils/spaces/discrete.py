from typing import Any, Dict

import torch
from adtool.utils.mutators import BaseMutator
from adtool.utils.spaces import BaseSpace


class DiscreteSpace(BaseSpace):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    /!\ mutation is gaussian by default: please create custom space inheriting from discrete space for custom mutation functions

    Example::

        >>> DiscreteSpace(2)

    """

    def __init__(self, n: int, mutator: BaseMutator = None, indpb: float = 1.0) -> None:
        """
        Init the elements useful to the spaces

        Args:
            n: discret value
            mutator: current mutator method
            indpb: Independent probability for each attribute to be exchanged
        """
        assert n >= 0
        self._n = n
        self._indpb = indpb

        super(DiscreteSpace, self).__init__((), torch.int64, mutator)

    def initialize(self, parent_obj: object) -> None:
        """
        Initialize the space.

        Args:
            parent_obj: The current autodisc module
        """
        # Apply potential binding
        super().initialize(parent_obj)
        self.n = self.apply_binding_if_existing(self._n, parent_obj)

        # indpb – independent probability for each attribute to be mutated.
        self.indpb = torch.as_tensor(self._indpb, dtype=torch.float64)

    def sample(self) -> torch.Tensor:
        """
        Generates a single random sample, upper than 0 and lower than n.

        Return:
            The return value is torch of the sample value
        """
        return torch.randint(self.n, ())

    def mutate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the mutation of the mutator on x

        Args:
            x: The variable to mutate
        Returns:
            x: the result of mutation
        """
        if self.mutator:
            mutate_mask = torch.rand(self.shape) < self.indpb
            x = self.mutator(x, mutate_mask)
            x = torch.floor(x).type(self.dtype)
            if not self.contains(x):
                return self.clamp(x)
            else:
                return x
        else:
            return x

    def contains(self, x: torch.Tensor) -> bool:
        """
        Check if x is bounded between 0 and n

        Args:
            x: The value to check
        Returns:
            The return value is True if x is bounded by 0 and n False otherwise
        """
        if isinstance(x, int):
            as_int = x
        # integer or size 0
        elif not x.dtype.is_floating_point and (x.shape == ()):
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Set x to an acceptable value of the space
        Args:
            x: The value to set
        Returns:
            x: After being set
        """
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.n - 1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        """
        Give a string representation of the class's object

        Returns:
            The return value is a string which represent our object
        """
        return "DiscreteSpace(%d)" % self.n

    def __eq__(self, other: object) -> bool:
        """
        Check if the other object are equal to the current object

        Args:
            other: An object
        Returns:
            The return value is True if self and other are equal False otherwise
        """
        return isinstance(other, DiscreteSpace) and self.n == other.n

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the object into JSON

        Returns:
            The JSON of the object
        """
        dict = super().to_json()
        dict["n"] = self._n
        dict["indpb"] = self._indpb
        return dict
