from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from pydoc import locate as _locate

import numpy as np


@dataclass
class DefaultEmbeddingBuilder:
    def build(self, data: Dict[str, Any]) -> np.ndarray:
        return np.asarray(data.get("output", []), dtype=float).reshape(-1)


def build_embedding_builder(
    config: Optional[Dict[str, Any]],
) -> Callable[[Dict[str, Any]], np.ndarray]:
    if not config:
        default_builder = DefaultEmbeddingBuilder()
        return default_builder.build

    path = config.get("path")
    if not path:
        raise ValueError("embedding_builder must include a 'path'")

    builder_cls = _locate(path)
    if builder_cls is None:
        raise ValueError(f"Could not retrieve class from path: {path}.")

    builder_config = config.get("config", {})
    builder = builder_cls(**builder_config)

    if not hasattr(builder, "build"):
        raise ValueError("embedding_builder must expose a build(data) method")

    return builder.build
