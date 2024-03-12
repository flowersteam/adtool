import numbers

import torch
from auto_disc.legacy.utils.spaces import BaseSpace


class MultiDiscreteSpace(BaseSpace):
    """
    - The multi-discrete space consists of a series of discrete spaces with different number of possible instances in eachs
    - Can be initialized as

        MultiDiscreteSpace([ 5, 2, 2 ])

    """

    def __init__(self, nvec, mutator=None, indpb=1.0):
        """
        Init the elements useful to the spaces

        Args:
            nvec: vector of counts of each categorical variable
            mutator: current mutator method
            indpb: Independent probability for each attribute to be exchanged
        """
        self._nvec = nvec
        self._indpb = indpb

        super(MultiDiscreteSpace, self).__init__(nvec, torch.int64, mutator)

    def initialize(self, parent_obj):
        """
        Initialize the space.

        Args:
            parent_obj: The current autodisc module
        """
        # Apply potential binding
        super().initialize(parent_obj)
        self._nvec = self.apply_binding_if_existing(self._nvec, parent_obj)
        self.shape = self._nvec
        assert (torch.tensor(self._nvec) > 0).all(), "nvec (counts) have to be positive"
        self.nvec = torch.as_tensor(self._nvec, dtype=torch.int64)

        # indpb – independent probability for each attribute to be mutated.
        if isinstance(self._indpb, numbers.Number):
            self._indpb = torch.full(self.nvec.shape, self._indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(self._indpb, dtype=torch.float64)

    def sample(self) -> torch.Tensor:
        """
        Generates a single random sample, upper than 0 and lower than n.

        Return:
            The return value is torch type of the sample value
        """
        return (torch.rand(self.nvec.shape) * self.nvec).type(self.dtype)

    def mutate(self, x):
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

    def contains(self, x) -> bool:
        """
        Check if x is bounded between 0 and n

        Args:
            x: The value to check
        Returns:
            The return value is True if x is bounded by 0 and n False otherwise
        """
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def clamp(self, x) -> torch.Tensor:
        """
        Set x to an acceptable value of the space
        Args:
            x: The value to set
        Returns:
            x: After being set
        """
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(
            x, torch.as_tensor(self.nvec - 1, dtype=self.dtype, device=x.device)
        )
        return x

    def __repr__(self):
        """
        Give a string representation of the class's object

        Returns:
            The return value is a string which represent our object
        """
        return "MultiDiscreteSpace({})".format(self.nvec)

    def __eq__(self, other):
        """
        Check if the other object are equal to the current object

        Args:
            other: An object
        Returns:
            The return value is True if self and other are equal False otherwise
        """
        return isinstance(other, MultiDiscreteSpace) and torch.all(
            self.nvec == other.nvec
        )

    def to_json(self):
        """
        Convert the object into JSON

        Returns:
            The JSON of the object
        """
        dict = super().to_json()
        dict["nvec"] = self._nvec
        dict["indpb"] = self._indpb
        return dict
