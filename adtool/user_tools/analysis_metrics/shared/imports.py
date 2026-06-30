from adtool.utils.factory import resolve_dotted_object


def load_dotted_object(path: str):
    return resolve_dotted_object(path, object_name="analysis object")
