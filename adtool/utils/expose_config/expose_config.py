from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Optional
import sys

#from adtool.utils.leafutils.leafstructs.registration import _REGISTRATION

from annotated_types import BaseMetadata, Le, Ge, Gt, Lt, Annotated

from adtool.utils.leafutils.leafstructs.registration import _REGISTRATION

def annotated_metadatas_to_json(annotation: List[BaseMetadata]):
    json = {}
    for m in annotation:
        if isinstance(m, Le):
            json["le"] = m.le
        if isinstance(m, Ge):
            json["ge"] = m.ge
        if isinstance(m, Gt):
            json["gt"] = m.gt
        if isinstance(m, Lt):
            json["lt"] = m.lt
    return json

from enum import Enum

def export_config(cls):
    json = {}
    for k,v in cls.model_fields.items():
        json[k] = {
            "type": v.annotation.__name__,
        #   "required": field.required,
            "default": v.default if v.default is not None else None
        }
        if  isinstance(v.annotation, Enum):
            json[k]["enum"] =  map(lambda x: x.value, v.annotation)

        if v.metadata:
            json[k]["metadata"] = annotated_metadatas_to_json(v.metadata)

    return json





#same but with a decorator
def expose(cls):
    dict_config = export_config(
    cls.config_type
    
    )

    previous_init = cls.__init__

    cls.JSON_CONFIG = dict_config
    def __init__(self, *args, **kwargs):
        self.config=self.config_type(*args, **kwargs)

        previous_init(self)


    cls.__init__ = __init__

    sub = cls.__module__.split(".")
    current = _REGISTRATION
    for i in range(1, len(sub)-1):
        if sub[i] not in current:
            current[sub[i]] = {}
        current = current[sub[i]]
    if sub[-1] not in current:

        current[sub[-1]] = [cls.__name__] 
    else:
        if not isinstance(current[sub[-1]], list):
            raise Exception(f"Error: {sub[-1]} is not a list")
        current[sub[-1]].append(cls.__name__)


    print("_REGISTRATION", _REGISTRATION, file=sys.stderr)
    return cls



# class Expose:
#     def __init__(self, config_type):
#         self.config_type = config_type

#     def __call__(self, *args, **kwargs):
#         new_class = type(*args, **kwargs)
#         new_class.config_type = self.config_type

#         #add a new init
#         def __init__(self, *args, **kwargs):
#             self.config=self.config_type(*args, **kwargs)

#         new_class.__init__ = __init__

#         return new_class



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
            key_name = list(self._config_defn.keys())[0]
            if key_name in cls.CONFIG_DEFINITION:
                raise ValueError(f"Config option {key_name} already exists.")
            # update
            cls.CONFIG_DEFINITION= {**cls.CONFIG_DEFINITION, **self._config_defn}
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
    #int or None
    def int_handler(domain: List[Optional[int]]) -> Dict[str, Optional[int]]:
        cleaned_domain = {i for i in domain if i is not None}
        if len(cleaned_domain) == 0:
            return {"min": None, "max": None}
        return {"min": min(cleaned_domain), "max": max(cleaned_domain)}

    @staticmethod
    def float_handler(domain: List[Optional[float]]) -> Dict[str, Optional[float]]:
        cleaned_domain = {i for i in domain if i is not None}
        if len(cleaned_domain) == 0:
            return {"min": None, "max": None}
        return {"min": min(cleaned_domain), "max": max(cleaned_domain)}

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
