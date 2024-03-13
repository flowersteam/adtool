import typing
from typing import Type

import torch
from addict import Dict
from adtool.auto_disc.utils.spaces import BaseSpace


class DictSpace(BaseSpace):
    """
    A Dict dictionary of simpler spaces.

    Example usage:
    self.genome_space = spaces.DictSpace({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:
    self.nested_genome_space = spaces.DictSpace({
        'sensors':  spaces.DictSpace({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        'inner_state':spaces.DictSpace({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.DictSpace({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """

    def __init__(self, spaces=None, **spaces_kwargs) -> None:
        """
        Init the elements useful to the space. Define spaces as dict of spaces

        Args:
            spaces: All spaces to be used
        """
        assert (spaces is None) or (
            not spaces_kwargs
        ), "Use either DictSpace(spaces=dict(...)) or DictSpace(foo=x, bar=z)"
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, list):
            spaces = Dict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(
                space, BaseSpace
            ), "Values of the attrdict should be instances of gym.Space"
        # None for shape and dtype, since it'll require special handling
        super().__init__(None, None)

    def initialize(self, parent_obj: object) -> None:
        """
        Initialize all spaces in dict.

        Args:
            parent_obj: The current autodisc module
        """
        for _, space in self.spaces.items():
            space.initialize(parent_obj)

    def sample(self) -> typing.Dict[str, torch.tensor]:
        """
        Generates sample for each spaces according to the space

        Return:
            The return value is Dict of sample values
        """
        return Dict([(k, space.sample()) for k, space in self.spaces.items()])

    def mutate(self, x):
        """
        Apply the mutation for each spaces according to the space

        Return:
            The return value is Dict of mutates values
        """
        return Dict([(k, space.mutate(x[k])) for k, space in self.spaces.items()])

    def contains(self, x: Dict) -> bool:
        """
        Check if each value is correctly set in its space
        Args:
            x: The dict of spaces
        Returns:
            The return value is True if each item in x is included in its space False otherwise
        """
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def clamp(self, x: Dict) -> Dict:
        """
        For each element in dict set it to an acceptable value of the space
        Args:
            x: The dict of value value to set
        Returns:
            x: After being set
        """
        return Dict([(k, space.clamp(x[k])) for k, space in self.spaces.items()])

    def __getitem__(self, key):
        return self.spaces[key]

    def __setitem__(self, key, value):
        self.spaces[key] = value

    def __delitem__(self, key):
        del self.spaces[key]

    def __iter__(self):
        for key in self.spaces:
            yield key

    def __repr__(self):
        return (
            "DictSpace("
            + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )

    def __eq__(self, other):
        return isinstance(other, DictSpace) and self.spaces == other.spaces

    def __len__(self):
        return len(self.spaces)

    def to_json(self):
        """
        Convert the object into JSON

        Returns:
            The JSON of the object
        """
        dict = {}
        for key, space in self.spaces.items():
            dict[key] = space.to_json()
        return dict
