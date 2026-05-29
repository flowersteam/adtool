from pydoc import locate

def make_module(module_name, *args, **kwargs):
    module_path = kwargs.get("path")
    if not module_path:
        raise ValueError(
            f"Missing path for {module_name} class."
        )
    
    # remove path from kwargs before passing to module constructor
    kwargs.pop("path")
    
    module = locate(module_path)
    if module is None:
        raise ValueError(
            f"Could not retrieve {module_name} from path: {module_path}."
        )
    return module(*args, **kwargs)