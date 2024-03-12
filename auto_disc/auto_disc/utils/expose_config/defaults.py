from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
import dataclasses
from typing import Any, Callable, List, Optional

from addict import Dict
from auto_disc.auto_disc.utils.expose_config.expose_config import (
    ExposeConfig,
    _handlers,
)
import sys


def defaults(
    default: Any, domain: Optional[List[Any]] = None, min: Any = None, max: Any = None
):
    """
    The canonical constructor for the _DefaultSetting dataclass,
    means that we don't accidentally expose the expose_config method except
    by subclassing the Defaults class.
    """
    return _DefaultSetting(default, domain, min, max)


class Defaults:
    """This class is only here for namespacing purposes."""

    @classmethod
    def expose_config(cls) -> Callable:
        """
        This decorator allows exposed config parameters to be exposed via a
        dataclass that inherits from Defaults.
        """
        # manually convert the dataclass to a dict
        # because pre-Python 3.9, decorator syntax is limited
        config_dict = cls._dataclass_to_config_dict()

        # create ExposeConfig objects to chain decorate
        decoration_chain: List[ExposeConfig] = []
        cls._wrap_config_defns(config_dict, decoration_chain)

        # return a big function composition of the decorator function
        return _compose(*decoration_chain)

    @classmethod
    def _wrap_config_defns(cls, config_dict, decoration_chain) -> List[ExposeConfig]:
        """
        Takes dict of config definitions (i.e., the dict form of
        a _DefaultSetting) and converts it into a list of ExposeConfig objects
        """
        # iterate over the dict and create expose_config classes to chain
        for k, v in config_dict.items():
            # handle both styles of providing domain
            # this checks either the domain is not set or set to the default
            # value of None by not being constructed in _DefaultSetting
            if v.get("domain", None) is None:
                try:
                    v["domain"] = [v["min"], v["max"]]
                except KeyError:
                    raise ValueError(
                        "To expose a config, "
                        "you must provide either "
                        "a domain or min/max."
                    )

            # setting up the big function composition but
            # NOTE : it doesn't actually matter what order they're called in,
            # unless the config itself is malformed
            decoration_chain.append(
                ExposeConfig(
                    name=k, default=v["default"], domain=v["domain"], parent=v["parent"]
                )
            )
        return decoration_chain

    @classmethod
    def _dataclass_to_config_dict(cls) -> Dict:
        """
        This function takes a (possible recursive) Defaults dataclass and
        converts it to a (flat) dict of config definitions. It's therefore
        the caller's responsibility to avoid key collisions.
        """
        config_dict = {}

        # inner function to recurse through the dataclass
        def recurse(dc: type, parent: str):
            for k, v in dc.__dataclass_fields__.items():
                # unwrap from the Field object
                unwrap_v = v.default

                # recurse, noting that past the root level, all values are
                # fields, which are instances of objects and not classes
                if isinstance(unwrap_v, Defaults):
                    # resolve path of recursive modules
                    parent += "." + k
                    recurse(unwrap_v, parent)
                else:
                    # base case, simply load the config dict from Defaults obj
                    if k in config_dict:
                        raise ValueError(f"Config option {k} already exists.")
                    else:
                        # consider default_factory
                        if v.default_factory is not dataclasses.MISSING:
                            unwrap_v = v.default_factory()
                        
                        
                        print("unwrap_v", unwrap_v, file=sys.stderr)
                        config_dict[k] = asdict(unwrap_v)

                        # remove the leading "." from the parent
                        # in a recursive call
                        if len(parent) > 0 and parent[0] == ".":
                            parent = parent[1:]

                        config_dict[k]["parent"] = parent

        recurse(cls, "")
        print("config_dict", config_dict, file=sys.stderr  )
        return config_dict


def deconstruct_recursive_dataclass_instance(params: Defaults):
    """Deconstruct a dataclass instance of type Defaults into a flat dictionary
    of simple key-value pairs."""

    p_dict = {}

    dicts_to_merge = []

    def set_nondestructively(d: dict, k: str, v: Any):
        # NOTE: I think in theory this is redundant, as key name clobbering
        # should only occur during the merging of the nested dataclass into the
        # top-level
        if d.get(k, None) is not None:
            raise KeyError(
                f"Key {k} exists multiple times in the dataclass {params}, cannot flatten."
            )
        else:
            d[k] = v

    def recurse(dc: Defaults):
        for field in fields(dc):
            if isinstance(getattr(dc, field.name), _DefaultSetting):
                # if it's a _DefaultSetting, set by the defined default
                val = field.default.default
                set_nondestructively(p_dict, field.name, val)
            elif not is_dataclass(field.type):
                # if it's a primitive type, set by runtime value
                val = getattr(dc, field.name)
                set_nondestructively(p_dict, field.name, val)
            elif issubclass(field.type, Defaults) and _is_default_dataclass(dc):
                # if it's a nested dataclass which is default initialized,
                # set by this defined value
                nested_dict = deconstruct_recursive_dataclass_instance(field.default)
                dicts_to_merge.append(nested_dict)
            elif issubclass(field.type, Defaults) and not _is_default_dataclass(dc):
                # if it's a nested dataclass which is user initialized, recurse
                child_dc = getattr(dc, field.name)
                recurse(child_dc)
            else:
                raise Exception("This branch should be unreachable.")

    # called for side effects which modify p_dict
    recurse(params)

    # merge top-level p_dict with the nested dataclass dicts parsed,
    # raising exceptions if there are key conflicts
    for nd in dicts_to_merge:
        try:
            p_dict = {**p_dict, **nd}
        except KeyError as e:
            raise e

    return p_dict


def _is_default_dataclass(dc: Any):
    """Test if a dataclass instance is default-initialized.

    NOTE: This is a glorified type check, so a value initialization which is the
    same as the default value initialization will count as having overridden the
    default
    """
    return dc == dc.__class__()


# NOTE: I think this is useless
def _is_default_field(params: Defaults, field_name: str) -> bool:
    """Test if the dataclass field is initialized or left as
    default.

    NOTE: This is a glorified type check, so a value initialization which is the
    same as the default value initialization will count as having overridden the
    default
    """

    # if it's a dataclass which is not a Defaults, check if the
    # dataclass was default-initialized
    field_val = getattr(params, field_name)

    if not isinstance(field_val, _DefaultSetting) and is_dataclass(field_val):
        return _is_default_dataclass(field_val)
    # if it's a Defaults, return True
    elif isinstance(field_val, _DefaultSetting):
        return True
    # if it's any other type, return False, as this implies the user
    # must have overridden the default
    else:
        return False


# NOTE: I think this is useless
def _is_default_field_r(params: Defaults, field_name: str) -> bool:
    """Recursively test if the dataclass field is initialized or left as
    default."""

    def recurse(dc: Defaults, field_name_query: str):
        for field in fields(dc):
            # if match, test predicate
            if field.name == field_name_query:
                return _is_default_field(dc, field.name)
            # if find nested dataclass that's not _DefaultSetting, recurse into it
            elif is_dataclass(getattr(dc, field.name)) and not isinstance(
                getattr(dc, field.name), _DefaultSetting
            ):
                child_dc = getattr(dc, field.name)
                return recurse(child_dc, field_name_query)

    query_result = recurse(params, field_name)
    if query_result is not None:
        return query_result
    else:
        raise KeyError(f"Could not find field {field_name} in object {params}.")


@dataclass
class _DefaultSetting:
    default: Any
    domain: Optional[List[Any]]
    min: Any
    max: Any


def _compose(*functions):
    """Compose functions à la pipes in FP."""

    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg

    return inner
