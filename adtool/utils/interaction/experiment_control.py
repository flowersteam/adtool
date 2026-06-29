from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_CONTROL_FILENAME = "experiment_control.json"
DEFAULT_EXPERIMENT_PAUSE_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_GOAL_ZONE_RADIUS = 0.18
_UNSET = object()


def experiment_control_path(control_dir: str | Path) -> Path:
    return Path(control_dir) / EXPERIMENT_CONTROL_FILENAME


def default_goal_targeting() -> dict:
    return {
        "radius": DEFAULT_GOAL_ZONE_RADIUS,
        "zones": [],
        "placement_supported": False,
        "projection_method": "",
        "projection_axes": [],
        "message": "",
        "resolved": None,
    }


def default_experiment_control() -> dict:
    return {
        "paused": False,
        "updated_at": None,
        "goal_targeting": default_goal_targeting(),
    }


def read_experiment_control(control_dir: str | Path) -> dict:
    path = experiment_control_path(control_dir)
    if not path.exists():
        return default_experiment_control()

    try:
        with path.open("r") as handle:
            payload = json.load(handle)
    except Exception:
        return default_experiment_control()

    return {
        "paused": bool(payload.get("paused", False)),
        "updated_at": payload.get("updated_at"),
        "goal_targeting": {
            **default_goal_targeting(),
            **payload.get("goal_targeting", {}),
        },
    }


def write_experiment_control(
    control_dir: str | Path,
    paused: bool | object = _UNSET,
    goal_targeting: dict | None | object = _UNSET,
) -> dict:
    control_dir = Path(control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    path = experiment_control_path(control_dir)
    payload = read_experiment_control(control_dir)

    if paused is not _UNSET:
        payload["paused"] = bool(paused)
    if goal_targeting is not _UNSET:
        payload["goal_targeting"] = {
            **default_goal_targeting(),
            **(goal_targeting or {}),
        }

    payload = {
        **payload,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle)
    tmp_path.replace(path)
    return payload


def wait_if_experiment_paused(
    control_dir: str | Path,
    poll_interval_seconds: float = DEFAULT_EXPERIMENT_PAUSE_POLL_INTERVAL_SECONDS,
) -> None:
    while read_experiment_control(control_dir).get("paused", False):
        time.sleep(poll_interval_seconds)
