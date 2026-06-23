from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adtool.examples.visu.highlights.loader import load_highlight_export_context
from adtool.examples.visu.highlights.provider import DiscoveryHighlightProvider


def materialize_discovery_filters(
    discoveries_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root_path = Path(discoveries_root)
    highlight_context = load_highlight_export_context(config_path)
    provider = highlight_context.provider

    if provider is None:
        raise ValueError("No discovery highlight provider configured")

    updated_count = 0
    for discovery_path in sorted(root_path.rglob("discovery.json")):
        materialize_discovery_file(discovery_path, provider)
        updated_count += 1

    return {"updated_count": updated_count}


def materialize_discovery_file(
    discovery_path: str | Path,
    provider: DiscoveryHighlightProvider,
) -> dict[str, Any]:
    discovery_path = Path(discovery_path)
    with discovery_path.open() as handle:
        payload = json.load(handle)

    payload["filters"] = provider.compute_filters(payload)
    _write_discovery_json(discovery_path, payload)
    return payload


def _write_discovery_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle)
    tmp_path.replace(path)
