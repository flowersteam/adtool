from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MIME_TYPES = {
    "html": "text/html",
    "js": "text/javascript",
    "css": "text/css",
    "mjs": "text/javascript",
    "json": "application/json",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "svg": "image/svg+xml",
    "mp4": "video/mp4",
    "webm": "video/webm",
}

RECOMPUTE_DEBOUNCE_SECONDS = 10.0
RECOMPUTE_MIN_INTERVAL_SECONDS = 15.0
ONLINE_POINT_UPDATE_INTERVAL_SECONDS = 5.0
ONLINE_FULL_RECOMPUTE_INTERVAL_SECONDS = 30.0

DEFAULT_DISPLAY_LIMIT = 500
MIN_DISPLAY_LIMIT = 1
MAX_DISPLAY_LIMIT = 10000
DISPLAY_LIMIT_PRESETS = [250, 500, 1000, 1500, 2000]

DEFAULT_PROJECTION_METHOD = "umap"
DEFAULT_PROJECTION_AXES = (0, 1)
PROJECTION_METHODS = ["umap", "pca", "tsne", "axis"]

DEFAULT_STICKER_PREVIEW_WORLD_HEIGHT = 0.6
MIN_STICKER_PREVIEW_WORLD_HEIGHT = 0.5
MAX_STICKER_PREVIEW_WORLD_HEIGHT = 1.2

DEFAULT_RANDOM_ITERATIONS = 100
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class ServerConfig:
    discoveries: Path
    config_file: Path | None = None
    static_dir: Path = STATIC_DIR
    refresh: bool = False


@dataclass
class RuntimeState:
    display_limit: int = DEFAULT_DISPLAY_LIMIT
    projection_method: str = DEFAULT_PROJECTION_METHOD
    projection_axes: tuple[int, int] = DEFAULT_PROJECTION_AXES
    sticker_preview_world_height: float = DEFAULT_STICKER_PREVIEW_WORLD_HEIGHT
    last_recompute_time: float = 0.0
    online_update_state: object | None = None
    analysis_lock: threading.Lock = field(default_factory=threading.Lock)
    recompute_lock: threading.Lock = field(default_factory=threading.Lock)
