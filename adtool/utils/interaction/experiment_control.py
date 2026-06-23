from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_CONTROL_FILENAME = "experiment_control.json"
DEFAULT_EXPERIMENT_PAUSE_POLL_INTERVAL_SECONDS = 1.0


def experiment_control_path(control_dir: str | Path) -> Path:
    return Path(control_dir) / EXPERIMENT_CONTROL_FILENAME


def read_experiment_control(control_dir: str | Path) -> dict:
    path = experiment_control_path(control_dir)
    if not path.exists():
        return {
            "paused": False,
            "updated_at": None,
        }

    try:
        with path.open("r") as handle:
            payload = json.load(handle)
    except Exception:
        return {
            "paused": False,
            "updated_at": None,
        }

    return {
        "paused": bool(payload.get("paused", False)),
        "updated_at": payload.get("updated_at"),
    }


def write_experiment_control(control_dir: str | Path, paused: bool) -> dict:
    control_dir = Path(control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    path = experiment_control_path(control_dir)
    payload = {
        "paused": bool(paused),
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
