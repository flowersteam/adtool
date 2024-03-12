from typing import Any, Dict, Tuple, Union

import torch
from auto_disc.legacy.base_autodisc_module import BaseAutoDiscModule
from auto_disc.legacy.utils.mutators import BaseMutator
from auto_disc.legacy.utils.spaces.utils import ConfigParameterBinding


class BaseSpace(object):
    """
    Defines the init_space, genome_space and intervention_space of a system
    """

    def __init__(
        self,
        shape: Tuple[Union[ConfigParameterBinding, Any]] = None,
        dtype: torch.dtype = None,
        mutator: BaseMutator = None,
    ) -> None:
        """
        Init the elements useful to the spaces

        Args:
            shape: Space shape
            dtype: torch type
            mutator: current mutator method
        """
        self.shape = None if shape is None else tuple(shape)
        self.dtype = dtype
        self.mutator = mutator

    def initialize(self, parent_obj: BaseAutoDiscModule) -> None:
        """
        Initialize the space.

        Args:
            parent_obj: The current autodisc module
        """
        if self.shape is not None:
            new_shape = []
            for elem in self.shape:
                new_shape.append(int(self.apply_binding_if_existing(elem, parent_obj)))
            self.shape = tuple(new_shape)
            if self.mutator:
                self.mutator.init_shape(self.shape)

    def apply_binding_if_existing(
        self, var: Union[ConfigParameterBinding, Any], lookup_obj: object
    ) -> Any:
        """
        Get result of config parameter binding operation

        Args:
            var: the current variable processed
            lookup_obj: current autodisc module in which the binding was made
        Returns:
            value: The result of the operation binding
        """
        if isinstance(var, ConfigParameterBinding):
            value = var.__get__(lookup_obj)
        else:
            value = var

        return value

    def sample(self):
        """
        Randomly sample an element of this space.
        Can be uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def mutate(self, x):
        """
        Randomly mutate an element of this space.

        Args:
            x: The element to be mutated
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space

        Args:
            x: The var which must be part of the space
        """
        raise NotImplementedError

    def clamp(self, x):
        """
        Return a valid clamped value of x inside space's bounds

        Args:
            x: a value
        """
        raise NotImplementedError

    def __contains__(self, x) -> bool:
        """
        Return boolean specifying if x is a valid
        member of this space

        Args:
            x: The var which must be part of the space
        """
        return self.contains(x)

    def to_json(self) -> Dict:
        """
        Convert the space into JSON

        Returns:
            The return value are a json containing a readable shape
        """
        shape = []
        if self.shape is not None:
            for element in self.shape:
                if isinstance(element, ConfigParameterBinding):
                    shape.append(element.to_json())
                else:
                    shape.append(element)
        else:
            shape = None
        return {"shape": shape}
