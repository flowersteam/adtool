from __future__ import annotations

import sys


def ensure_adtool_examples_alias() -> None:
    try:
        import adtool
        import examples
    except ImportError:
        return

    sys.modules.setdefault("adtool.examples", examples)
    if not hasattr(adtool, "examples"):
        setattr(adtool, "examples", sys.modules["adtool.examples"])
