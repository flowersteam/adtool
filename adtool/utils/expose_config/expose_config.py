from enum import Enum
from typing import Any, Dict, List, Optional

from annotated_types import BaseMetadata, Le, Ge, Gt, Lt
from pydantic_core import to_jsonable_python

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


def export_config(cls):
    exported = {}
    for name, field in cls.model_fields.items():
        annotation = field.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        default = (
            None
            if field.is_required()
            else to_jsonable_python(field.get_default(call_default_factory=True))
        )
        exported[name] = {
            "type": type_name,
            "default": default,
        }
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            exported[name]["enum"] = [item.value for item in annotation]

        if field.metadata:
            exported[name]["metadata"] = annotated_metadatas_to_json(field.metadata)

    return exported


def expose(cls):
    dict_config = export_config(cls.config)

    previous_init = cls.__init__

    cls.JSON_CONFIG = dict_config

    def __init__(self, *args, **kwargs):
        self.config = self.config(*args, **kwargs)
        previous_init(self, **kwargs)

    cls.__init__ = __init__

    sub = cls.__module__.split(".")
    current = _REGISTRATION
    for i in range(1, len(sub) - 1):
        if sub[i] not in current:
            current[sub[i]] = {}
        current = current[sub[i]]
    if sub[-1] not in current:
        current[sub[-1]] = [cls.__name__]
    else:
        if not isinstance(current[sub[-1]], list):
            raise Exception(f"Error: {sub[-1]} is not a list")
        current[sub[-1]].append(cls.__name__)

    return cls
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
