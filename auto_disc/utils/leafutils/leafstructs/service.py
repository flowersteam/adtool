from copy import deepcopy
from functools import partial
from typing import Any, List

from auto_disc.utils.leaf.Leaf import Leaf


def provide_leaf_as_service(
    object: Any, leaf_cls: Leaf, overridden_attr: List["str"] = None
) -> Any:
    if overridden_attr is None:
        overridden_attr = []
        for k in leaf_cls.__dict__.keys():
            # fetch all public methods
            if k[0] != "_":
                overridden_attr.append(k)

    # override methods in mutable state dict of object
    for name in overridden_attr:
        new_attr = getattr(leaf_cls, name)

        if callable(new_attr):
            # need __get__ here to bound the method from the class to the right object
            setattr(object, name, new_attr.__get__(object))
        else:
            setattr(object, name, new_attr)

    return object
