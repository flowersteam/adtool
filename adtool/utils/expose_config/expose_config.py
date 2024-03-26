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


import inspect


#same but with a decorator
def expose(cls):
    dict_config = export_config(
    cls.config
    
    )

    print("expose", cls, cls.__module__, file=sys.stderr)

    previous_init = cls.__init__

    #get signature of the init
    signature = inspect.signature(cls.__init__)
    print("signature", signature, file=sys.stderr)

    cls.JSON_CONFIG = dict_config
    def __init__(self, *args, **kwargs):
        print("CURRENT SELF", self, file=sys.stderr)
        self.config=self.config(*args, **kwargs)

        print("CURRENT ARGS", kwargs, file=sys.stderr)

        previous_init(self, **kwargs)


        


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

