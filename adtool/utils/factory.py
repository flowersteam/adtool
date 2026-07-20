from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pydoc import locate
from typing import Any, Generic, TypeVar, overload

from pydantic import BaseModel, ConfigDict

ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class ObjectSpec(Generic[ConfigT]):
    """Import path and keyword arguments used to construct an object."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    path: str
    config: ConfigT = field(default_factory=dict)


def class_path_of(cls: type) -> str:
    """Return the fully qualified import path for ``cls``."""

    qualified_class_name = cls.__qualname__
    module_name = cls.__module__
    return module_name + "." + qualified_class_name


@overload
def object_spec(path: str) -> ObjectSpec[dict[str, Any]]: ...


@overload
def object_spec(path: str, config: ConfigT) -> ObjectSpec[ConfigT]: ...


def object_spec(
    path: str,
    config: Any = None,
) -> ObjectSpec[Any]:
    """Build an :class:`ObjectSpec` while copying its configuration."""

    if config is None:
        normalized_config = {}
    elif isinstance(config, BaseModel):
        normalized_config = config.model_dump()
    elif isinstance(config, Mapping):
        normalized_config = dict(config)
    else:
        raise TypeError("Object spec config must be a mapping or Pydantic model.")
    return ObjectSpec(path=path, config=normalized_config)


def coerce_object_spec(
    raw: ObjectSpec[Any] | Mapping[str, Any] | Any,
    *,
    object_name: str = "object",
) -> ObjectSpec[dict[str, Any]]:
    """Validate and normalize the canonical ``path``/``config`` shape."""

    if isinstance(raw, ObjectSpec) or (
        hasattr(raw, "path") and hasattr(raw, "config")
    ):
        raw = {
            "path": getattr(raw, "path"),
            "config": getattr(raw, "config"),
        }

    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{object_name.capitalize()} spec must be a mapping with 'path' and 'config'."
        )

    if "config" not in raw:
        raise ValueError(
            f"{object_name.capitalize()} spec requires a 'config' mapping."
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

    config = raw["config"]
    if isinstance(config, BaseModel):
        config = config.model_dump()
    elif not isinstance(config, Mapping):
        raise TypeError(
            f"{object_name.capitalize()} spec 'config' must be a mapping "
            "or Pydantic model."
        )

    return ObjectSpec(path=path.strip(), config=dict(config))


def resolve_dotted_object(path: str, *, object_name: str = "object") -> Any:
    """Resolve an importable dotted path or raise a contextual error."""

    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{object_name.capitalize()} path must be a non-empty string.")

    obj = locate(path)
    if obj is None:
        raise ValueError(f"Could not resolve {object_name}: {path}")

    return obj


def instantiate_object(
    spec: ObjectSpec[Any] | Mapping[str, Any] | Any,
    *args: Any,
    object_name: str = "object",
    **config_overrides: Any,
) -> Any:
    """Instantiate an object from a canonical spec and optional overrides."""

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
