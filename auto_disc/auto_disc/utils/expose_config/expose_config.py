from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Optional


class ExposeConfig:
    def __init__(self, *args, **kwargs) -> None:
        self._config_defn = self._generate_config(*args, **kwargs)

    @staticmethod
    def _generate_config(
        name: str,
        default: Any,
        domain: Optional[List[Any]] = None,
        parent: Optional[str] = None,
    ) -> Dict[str, Any]:
        # convert Python type names to the corresponding config type names
        exposed_type = type(default)
        exposed_type_name = _python_type_to_config_type[exposed_type.__name__.upper()]

        # convert the domain specification to the proper key-value pairs
        domain_entry = _handle_type(exposed_type, domain)

        # construct the config definition
        config_defn = {
            name: {
                "type": exposed_type_name,
                "default": default,
                "parent": parent,
                **domain_entry,
            }
        }

        return config_defn

    def __call__(self, cls: type) -> Any:
        # upsert CONFIG_DEFINITION into the class
        if not hasattr(cls, "CONFIG_DEFINITION"):
            # insert
            cls.CONFIG_DEFINITION = self._config_defn
        else:
            # warn of key collision
            key_name = list(self._config_defn.keys())[0]
            if key_name in cls.CONFIG_DEFINITION:
                raise ValueError(f"Config option {key_name} already exists.")
            # update
            cls.CONFIG_DEFINITION.update(self._config_defn)
        return cls


class expose_config(ExposeConfig):
    """
    This is the decorator that should be used to expose config options.
    It's just a factory for ExposeConfig.
    """

    pass


def _handle_type(type: type, domain: Optional[List[Any]]) -> Dict[str, Any]:
    """
    Converts the data in `domain` to the proper key-value pairs for inclusion
    in the CONFIG_DEFINITION.
    """
    py_type_name = type.__name__.upper()
    type_handler = _handlers[_python_type_to_config_type[py_type_name]]
    return type_handler(domain)


class Handlers:
    """Namespace for handlers"""

    @staticmethod
    def bool_handler(domain: Optional[List[bool]] = None) -> Dict[str, List[bool]]:
        # ignores domain, as it is always [True, False]
        return {"possible_values": [True, False]}

    @staticmethod
    def int_handler(domain: List[int]) -> Dict[str, List[int]]:
        return {"min": min(domain), "max": max(domain)}

    @staticmethod
    def float_handler(domain: List[float]) -> Dict[str, List[float]]:
        return {"min": min(domain), "max": max(domain)}

    @staticmethod
    def str_handler(domain: List[str]) -> Dict[str, List[str]]:
        return {"possible_values": domain}

    @staticmethod
    def dict_handler(domain: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        # NOTE: this should never be used due to mutability of dicts
        return {"possible_values": domain}


_handlers = {
    "BOOLEAN": Handlers.bool_handler,
    "INTEGER": Handlers.int_handler,
    "DECIMAL": Handlers.float_handler,
    "STRING": Handlers.str_handler,
    "DICT": Handlers.dict_handler,
}

_python_type_to_config_type = {
    "BOOL": "BOOLEAN",
    "INT": "INTEGER",
    "FLOAT": "DECIMAL",
    "STR": "STRING",
    "DICT": "DICT",
}


# def _update(d, u):
#     """
#     Small function for recursively updating a dict, taken from
#     https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

#     """
#     for k, v in u.items():
#         if isinstance(v, dict):
#             d[k] = _update(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d
