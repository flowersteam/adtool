#!/usr/bin/env python3

"""A "build" script to add custom modules to adtool.

After adding your code to the appropriate folders, you can simply run `build.py`
to ensure any dependencies of your code that are under `adtool_custom` as a
Poetry package are also appropriately installed.

For example, given a correct installation like:
    adtool_custom/
    ├─ systems/
    │  ├─ MySystem.py
    │  ├─ my_system/
    │  │  ├─ pyproject.toml
    │  │  ├─ ...
Running build.py will install the Poetry project found at
    `adtool_custom/systems/my_system`
as a develop dependency (similar to `pip -e install .`)
"""
import os
from typing import Callable

import toml


def inject_path_dependency(pyproject_path: str, injected_path_dep: str) -> None:
    """Injects `injected_path_dep` as a path dependency in the Poetry project
    with pyproject.toml given by `pyproject_path`."""
    with open(pyproject_path, "r") as f:
        parsed_toml = toml.loads(f.read())

    # TODO: parse the injected_path_dep to get the properly declared pkg name
    # instead of doing this
    pkg_name = injected_path_dep.split(os.sep)[-1]

    # load in the adtool_custom module too
    # FIXME: do this in a different function
    parsed_toml["tool"]["poetry"]["packages"].append({"include": "adtool_custom"})
    # injection
    parsed_toml["tool"]["poetry"]["dependencies"][pkg_name] = {
        "path": injected_path_dep,
        "develop": True,
    }

    with open(pyproject_path, "w") as f:
        # overwrite old
        toml.dump(parsed_toml, f)


def get_packages_in_place(relative_dirpath: str) -> list[str]:
    """Search directory and return what Python packages/modules are present."""
    package_list = []

    def append_package_fn_on_predicate(dir):
        if is_poetry_pkg_dir(dir):
            package_list.append(dir)

    map_over_subdirs(append_package_fn_on_predicate, relative_dirpath)
    return package_list


def map_over_subdirs(f: Callable, relative_dirpath: str, max_depth: int = 2) -> None:
    for root, dirs, _ in os.walk(relative_dirpath, topdown=True):
        path = root.split(os.sep)
        depth = len(path) - 2

        if depth > max_depth:
            break

        # with topdown=True, we can prune the search
        # NOTE: use slice notation to modify in place
        #       needed for os.walk, see its documentation
        dirs[:] = list(filter(lambda x: x != "__pycache__", dirs))

        f(root)
    return


def is_poetry_pkg_dir(dirpath: str) -> bool:
    """Check if the provided directory contains pyproject.toml, which we use as
    a heuristic for being a Poetry package."""
    os.listdir(dirpath)
    return "pyproject.toml" in os.listdir(dirpath)


def main() -> None:
    try:
        package_list = get_packages_in_place("./adtool_custom")
        for pkg in package_list:
            inject_path_dependency("./pyproject.toml", pkg)
        print("Success! Remember to run poetry install again.")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
