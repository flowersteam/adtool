from enum import Enum
from typing import List

from annotated_types import BaseMetadata, Le, Ge, Gt, Lt
from pydantic_core import to_jsonable_python

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

    return cls
