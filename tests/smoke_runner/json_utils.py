"""JSON and parsing helpers for smoke-test specs and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_file(path: Path) -> Any:
    """Read and parse a JSON file with actionable errors."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Unable to read JSON file: {path}") from exc

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in file: {path}") from exc


def parse_string_list(value: Any, field_name: str, file_path: Path) -> list[str]:
    """Parse a list of strings from a JSON field."""
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            f"Field '{field_name}' in {file_path} must be a list of strings"
        )

    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"Field '{field_name}' in {file_path} must contain only strings"
            )
        stripped = item.strip()
        if stripped:
            parsed.append(stripped)

    return parsed
