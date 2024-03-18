"""
Helper functions for doing namespace resolution. See `test_registration.py` for usage demonstration.
NOTE: Will be deprecated later.
"""
import importlib
import math
import pkgutil
from pydoc import locate as _locate
from typing import cast
import sys

from mergedeep import merge





# legacy compatibility
_REGISTRATION: dict[str, dict] = {
    "systems": {},
    "explorers": {
        "IMGEPExplorer": "adtool.explorers.IMGEPFactory",
    },
    "maps": {},
    "input_wrappers": {
        "generic.CPPN": "adtool.input_wrappers.generic.cppn.cppn_input_wrapper.CppnInputWrapper",
    },
    "output_representations": {
        "specific.LeniaFlattenImage": "adtool.output_representations.specific.lenia_output_representation.LeniaImageRepresentation",
        "specific.LeniaStatistics": "adtool.output_representations.specific.lenia_output_representation_hand_defined.LeniaHandDefinedRepresentation",
    },
}


def locate_cls(cls_path: str) -> type:
    """
    Wrapper function which allows static type checking by providing the correct
    function signature.
    """
    return cast(type, _locate(cls_path))


def get_path_from_cls(cls: type) -> str:
    """
    Returns the fully qualified class path, for use with dynamic imports.
    """
    # TODO: refactor this function out into it's own function, as `Leaf.py`
    # depends on it but definitely ought not to...
    qualified_class_name = cls.__qualname__
    module_name = cls.__module__
    class_path = module_name + "." + qualified_class_name
    return class_path


def get_cls_from_path(cls_path: str) -> type:
    """
    Returns the class pointed to by a fully qualified class path,
    importing along the way.
    """
    return locate_cls(cls_path)


def get_cls_from_name(cls_name: str, ad_type_name: str) -> type:
    """
    Attempts to retrieve the class by solely its name and the "type" of object
    it is (among `explorers`, `systems`, etc.), looking up the
    information along the way.
    """
    if "." not in ad_type_name:
        # for all ad_type_name that are not callbacks
        type_lookup_info = get_modules(ad_type_name)
    else:
        # for callbacks
        callback_type_arr = ad_type_name.split(".")
        assert callback_type_arr[0] == "callbacks"
        callback_lookup_dict = get_modules(callback_type_arr[0])
        type_lookup_info = callback_lookup_dict[callback_type_arr[1]]

    
    cls_path = type_lookup_info[cls_name]
    return locate_cls(cls_path)


def get_submodules(submodule: str, namespace: str) -> dict[str, str]:
    """Get dict of submodule fully-qualified names from the submodule type and
    its namespace.

    NOTE: By our convention, only uppercased Python modules will be walked.
    This allows us to retrieve classes, as Python has no public/private
    distinction. The user should name any library functions and other code
    with lowercased names, and the user shall not uppercase anything except
    that which they desire to import into the software.
    """

    # for clarity, `adtool_module` is a module in the context of the software,
    # to disambiguate from Python modules, which is what `pkgutil` expects

    lookup_prefix = namespace + "." + submodule
    module = importlib.import_module(lookup_prefix)
    path_prefix = lookup_prefix + "."
    it = pkgutil.iter_modules(module.__path__, prefix=path_prefix)

    adtool_module_dict = {}
    for el in it:
        adtool_module_shortname = el.name.split(".")[-1]
        # filter to only relevant classes
        if adtool_module_shortname[0].isupper():
            adtool_module_fqpath = el.name + "." + adtool_module_shortname
            adtool_module_dict[adtool_module_shortname] = adtool_module_fqpath
        else:
            # TODO: logging here
            pass

    return adtool_module_dict




def get_examples_modules(submodule: str) -> dict[str, str]:
    
    return get_submodules(submodule, namespace="examples")


def get_legacy_modules(submodule: str) -> dict[str, str]:
    # NOTE: we only return the fully qualified module path instead of
    # module itself to avoid polluting the namespace with imports

    # TODO: don't hardcode this
    return _REGISTRATION.get(submodule)


def get_modules(submodule_type: str) -> dict:
    """Get dict of submodule fully-qualified names from the submodule type and
    its namespace.

    #### Args
    - submodule_type (str): The type of submodule, chosen within
        {systems, explorers, maps, callbacks}.

    #### Returns
    A dictionary returning with items such as
        {"SystemName" : "adtool.systems.SystemName.SystemName"}

    """

    result= dict(
            merge(
                # because merge mutates the first arg
                {},
                get_legacy_modules(submodule_type),
      #          get_default_modules(submodule_type),
                get_examples_modules(submodule_type),
              #  get_custom_modules(submodule_type),
            )
        )
    print("result", result, file=sys.stderr)
    return result


def _check_jsonify(my_dict):
    try:
        for key, value in my_dict.items():
            if isinstance(my_dict[key], dict):
                _check_jsonify(my_dict[key])
            elif isinstance(my_dict[key], list):
                for x in my_dict[key]:
                    if isinstance(my_dict[key], dict):
                        _check_jsonify(x)
                    elif isinstance(my_dict[key], float):
                        if math.isinf(my_dict[key]):
                            if my_dict[key] > 0:
                                my_dict[key] = "inf"
                            else:
                                my_dict[key] = "-inf"
            elif isinstance(my_dict[key], float):
                if math.isinf(my_dict[key]):
                    if my_dict[key] > 0:
                        my_dict[key] = "inf"
                    else:
                        my_dict[key] = "-inf"
    except Exception as ex:
        print(
            "my_dict = ",
            my_dict,
            "key = ",
            key,
            "my_dict[key] = ",
            my_dict[key],
            "exception =",
            ex,
        )


def get_adtool_registered_modules_info(registered_modules):
    infos = []
    for module_name, module_class_path in registered_modules.items():
        module_class = get_cls_from_path(module_class_path)
        if module_class is None:
            raise ValueError(
                f"Could not retrieve class from path: {module_class_path}."
            )
        info = {}
        info["name"] = module_name
        try:
            info["config"] = module_class.CONFIG_DEFINITION
        except AttributeError:
            raise ValueError(f"Could not load class: {module_class_path}")
        if hasattr(module_class, "input_space"):
            info["input_space"] = module_class.input_space.to_json()

        if hasattr(module_class, "output_space"):
            info["output_space"] = module_class.output_space.to_json()

        if hasattr(module_class, "step_output_space"):
            info["step_output_space"] = module_class.step_output_space.to_json()
        _check_jsonify(info)
        infos.append(info)
    return infos


def get_adtool_registered_callbacks(registered_callbacks):
    infos = []
    for category, callbacks_list in registered_callbacks.items():
        info = {}
        info["name_callbacks_category"] = category
        info["name_callbacks"] = list(callbacks_list.keys())
        infos.append(info)
    return infos
