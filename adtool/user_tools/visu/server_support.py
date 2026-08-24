from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback
from typing import Any

from fastapi import HTTPException

from .runtime import MIME_TYPES


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def mime_type(path: str | Path) -> str:
    extension = str(path).rsplit(".", maxsplit=1)[-1].lower()
    return MIME_TYPES.get(extension, "application/octet-stream")


def resolve_input_path(value: Any, field_name: str, required: bool = True) -> Path | None:
    if value is None or str(value).strip() == "":
        if required:
            raise HTTPException(status_code=422, detail=f"{field_name} is required.")
        return None

    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def require_directory(path: Path, field_name: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{field_name} does not exist: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=422, detail=f"{field_name} must be a directory: {path}")


def require_file(path: Path, field_name: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise HTTPException(status_code=422, detail=f"{field_name} must be a file: {path}")


def payload_int(
    payload: dict[str, Any],
    field_name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = payload.get(field_name, default)
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{field_name} must be an integer.")

    if minimum is not None and parsed < minimum:
        raise HTTPException(status_code=422, detail=f"{field_name} must be at least {minimum}.")
    if maximum is not None and parsed > maximum:
        raise HTTPException(status_code=422, detail=f"{field_name} must be at most {maximum}.")
    return parsed


def optional_payload_int(payload: dict[str, Any], field_name: str) -> int | None:
    if field_name not in payload or payload.get(field_name) in (None, ""):
        return None
    return payload_int(payload, field_name, 0, minimum=2)


def timestamped_analysis_dir(discoveries_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (discoveries_dir / "analysis_runs" / f"{prefix}_{timestamp}").resolve()


def error_detail(prefix: str, exc: Exception) -> str:
    """Return an actionable client-side error that includes the full traceback.

    The visualisation server is a local development tool, so surfacing the
    traceback to the browser is considerably more useful than replacing it
    with an opaque HTTP 500 response.  Callers also log the same exception on
    the server side.
    """
    message = str(exc) or exc.__class__.__name__
    formatted_traceback = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).rstrip()
    return f"{prefix}: {message}\n\nServer traceback:\n{formatted_traceback}"
