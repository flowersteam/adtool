from pydoc import locate


def load_dotted_object(path: str):
    obj = locate(path)
    if obj is None:
        raise ValueError(f"Could not import {path}")
    return obj
