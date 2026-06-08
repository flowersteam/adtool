from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

if __package__:
    from .runtime import ServerConfig
    from .server_support import is_relative_to
else:
    from runtime import ServerConfig
    from server_support import is_relative_to


def _export_source_dir(config: ServerConfig, file_path: str) -> Path | None:
    normalized = file_path.lstrip("/")
    if normalized.startswith("discoveries/"):
        normalized = normalized[len("discoveries/"):]

    relative_dir = Path(normalized).parent
    if relative_dir == Path("."):
        return None

    source_dir = (config.discoveries / relative_dir).resolve()
    if not is_relative_to(source_dir, config.discoveries):
        return None
    if not source_dir.exists() or not source_dir.is_dir():
        return None
    return source_dir


def export_selected_discoveries(
    config: ServerConfig,
    files: Iterable[str],
) -> dict[str, str]:
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    export_dir = (config.discoveries.parent / current_time).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    copied_dirs = set()
    for file_path in files:
        source_dir = _export_source_dir(config, file_path)
        if source_dir is None or source_dir in copied_dirs:
            continue

        copied_dirs.add(source_dir)
        destination = export_dir / source_dir.name
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)

    return {"status": "ok", "new_dir": str(export_dir)}
