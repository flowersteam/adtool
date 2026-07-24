"""Mutation strategy delegated to a parameter map."""

from __future__ import annotations

from typing import Any

from adtool.mutators.base import BaseMutator


class SpecificMutator(BaseMutator):
    """Delegate mutation to ``parameter_map.mutate``.

    This is the standard IMGEP mutator for structured parameter dictionaries.
    The parameter map is supplied by the explorer at invocation time, keeping
    this object independent and directly pickleable.
    """

    def __call__(self, parameters: Any, *, parameter_map: Any = None) -> Any:
        if parameter_map is None or not hasattr(parameter_map, "mutate"):
            raise TypeError(
                "SpecificMutator requires a parameter map with a mutate method."
            )
        return parameter_map.mutate(parameters)
