from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from adtool.utils.factory import instantiate_object

from .provider import DiscoveryHighlightProvider


def empty_highlight_schema() -> dict[str, object]:
    return {"fields": [], "rules": [], "filters_detected": False, "storage_key": ""}


@dataclass
class HighlightExportContext:
    provider: DiscoveryHighlightProvider | None
    schema: dict


def load_highlight_export_context(
    config_path: str | Path | None,
) -> HighlightExportContext:
    provider_config = _load_provider_config(config_path)
    if provider_config is None:
        return HighlightExportContext(provider=None, schema=empty_highlight_schema())

    provider = instantiate_object(
        provider_config,
        object_name="discovery highlight provider",
    )
    return HighlightExportContext(provider=provider, schema=provider.schema())


def _load_provider_config(config_path: str | Path | None) -> dict | None:
    if config_path is None:
        return None

    with Path(config_path).open() as handle:
        config = json.load(handle)

    return config.get("discovery_highlights")
