from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pydoc import locate
from typing import Any


@dataclass(frozen=True)
class ObjectSpec:
    path: str
    config: dict[str, Any] = field(default_factory=dict)


def class_path_of(cls: type) -> str:
    qualified_class_name = cls.__qualname__
    module_name = cls.__module__
    return module_name + "." + qualified_class_name


def object_spec(
    path: str,
    config: Mapping[str, Any] | None = None,
) -> ObjectSpec:
    return ObjectSpec(path=path, config=dict(config or {}))


def coerce_object_spec(
    raw: ObjectSpec | Mapping[str, Any] | Any,
    *,
    object_name: str = "object",
) -> ObjectSpec:
    if isinstance(raw, ObjectSpec):
        return raw

    if hasattr(raw, "path") and hasattr(raw, "config"):
        raw = {
            "path": getattr(raw, "path"),
            "config": getattr(raw, "config"),
        }

    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{object_name.capitalize()} spec must be a mapping with 'path' and 'config'."
        )

    extra_keys = sorted(set(raw.keys()) - {"path", "config"})
    if extra_keys:
        extras = ", ".join(extra_keys)
        raise ValueError(
            f"{object_name.capitalize()} spec contains unsupported keys: {extras}. "
            "Use the canonical {'path': ..., 'config': {...}} shape."
        )

    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(
            f"{object_name.capitalize()} spec requires a non-empty string 'path'."
        )

    config = raw.get("config", {})
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError(
            f"{object_name.capitalize()} spec 'config' must be a mapping."
        )

    return ObjectSpec(path=path.strip(), config=dict(config))


def resolve_dotted_object(path: str, *, object_name: str = "object") -> Any:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{object_name.capitalize()} path must be a non-empty string.")

    obj = locate(path)
    if obj is None:
        raise ValueError(f"Could not resolve {object_name}: {path}")

    return obj


def instantiate_object(
    spec: ObjectSpec | Mapping[str, Any] | Any,
    *args: Any,
    object_name: str = "object",
    **config_overrides: Any,
) -> Any:
    normalized = coerce_object_spec(spec, object_name=object_name)
    factory = resolve_dotted_object(normalized.path, object_name=object_name)
    kwargs = dict(normalized.config)
    kwargs.update(config_overrides)
    try:
        return factory(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Could not instantiate {object_name} '{normalized.path}' with config "
            f"keys: {sorted(kwargs.keys())}"
        ) from exc
