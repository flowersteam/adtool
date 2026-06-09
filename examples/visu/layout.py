from __future__ import annotations

import time
from pathlib import Path

from watchfiles import Change, watch

if __package__:
    from .runtime import (
        RECOMPUTE_DEBOUNCE_SECONDS,
        RECOMPUTE_MIN_INTERVAL_SECONDS,
        RuntimeState,
        ServerConfig,
    )
else:
    from runtime import (
        RECOMPUTE_DEBOUNCE_SECONDS,
        RECOMPUTE_MIN_INTERVAL_SECONDS,
        RuntimeState,
        ServerConfig,
    )


def _compute_coordinates(*args, **kwargs) -> None:
    if __package__:
        from .coordinates import compute_coordinates
    else:
        from coordinates import compute_coordinates

    compute_coordinates(*args, **kwargs)


def cleanup_static_discoveries(config: ServerConfig) -> None:
    (config.static_dir / "discoveries.json").unlink(missing_ok=True)
    for atlas_path in config.static_dir.glob("discovery_atlas_*.png"):
        atlas_path.unlink(missing_ok=True)


def write_discovery_coordinates(config: ServerConfig, state: RuntimeState) -> None:
    _compute_coordinates(
        config.discoveries,
        static_dir=config.static_dir,
        max_displayed=state.display_limit,
        projection_method=state.projection_method,
        projection_axes=state.projection_axes,
    )


def recompute_discoveries(
    config: ServerConfig,
    state: RuntimeState,
    ignore_interval: bool = False,
    respect_interval: bool = False,
) -> bool:
    with state.recompute_lock:
        now = time.monotonic()
        if (
            respect_interval
            and not ignore_interval
            and now - state.last_recompute_time < RECOMPUTE_MIN_INTERVAL_SECONDS
        ):
            return False

        write_discovery_coordinates(config, state)
        state.last_recompute_time = time.monotonic()
        return True


def is_relevant_discovery_change(changes: set[tuple[Change, str]]) -> bool:
    watched_names = {"discovery.json", "config.json"}
    watched_suffixes = {".png", ".mp4"}

    return any(
        change in (Change.added, Change.modified, Change.deleted)
        and (
            Path(path).name in watched_names
            or Path(path).suffix.lower() in watched_suffixes
        )
        for change, path in changes
    )


def watch_discoveries(config: ServerConfig, state: RuntimeState) -> None:
    print("Watching discoveries")
    for changes in watch(config.discoveries, recursive=True):
        if not is_relevant_discovery_change(changes):
            continue

        print("Change in discoveries")
        time.sleep(RECOMPUTE_DEBOUNCE_SECONDS)
        try:
            if recompute_discoveries(config, state, respect_interval=True):
                print("Discoveries recomputed")
            else:
                print("Discovery recompute skipped: waiting for live-update interval")
        except ValueError as error:
            print(f"Discovery recompute failed: {error}")
