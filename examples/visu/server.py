from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from watchfiles import Change, awatch, watch

MIME_TYPES = {
    "html": "text/html",
    "js": "text/javascript",
    "css": "text/css",
    "mjs": "text/javascript",
    "json": "application/json",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "mp4": "video/mp4",
    "webm": "video/webm",
}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RECOMPUTE_DEBOUNCE_SECONDS = 10.0
RECOMPUTE_MIN_INTERVAL_SECONDS = 15.0
DEFAULT_DISPLAY_LIMIT = 500
MIN_DISPLAY_LIMIT = 1
MAX_DISPLAY_LIMIT = 10000
DISPLAY_LIMIT_PRESETS = [250, 500, 1000, 1500, 2000]


@dataclass(frozen=True)
class ServerConfig:
    discoveries: Path
    static_dir: Path = STATIC_DIR
    coverage_run: Path | None = None
    refresh: bool = False


@dataclass
class RuntimeState:
    display_limit: int = DEFAULT_DISPLAY_LIMIT
    last_recompute_time: float = 0.0
    recompute_lock: threading.Lock = field(default_factory=threading.Lock)


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discoveries", type=str, required=True)
    parser.add_argument("--coverage_run", type=str, required=False, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    return ServerConfig(
        discoveries=Path(args.discoveries).resolve(),
        coverage_run=Path(args.coverage_run).resolve() if args.coverage_run else None,
        refresh=args.refresh,
    )


def _compute_coordinates(*args, **kwargs) -> None:
    try:
        from .coordinates import compute_coordinates
    except ImportError:
        from coordinates import compute_coordinates

    compute_coordinates(*args, **kwargs)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _mime_type(path: str | Path) -> str:
    extension = str(path).rsplit(".", maxsplit=1)[-1].lower()
    return MIME_TYPES.get(extension, "application/octet-stream")


def _coverage_roots(config: ServerConfig) -> list[Path]:
    if config.coverage_run is None:
        return []

    roots = [config.coverage_run]
    if config.coverage_run.name.startswith("coverage_run_"):
        roots.append(config.coverage_run.parent)

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def _latest_summary_under(root: Path) -> Path | None:
    direct_summary = root / "summary.json"
    if direct_summary.exists():
        return direct_summary

    summaries = [
        summary
        for summary in root.glob("coverage_run_*/summary.json")
        if summary.is_file()
    ]
    if not summaries:
        return None

    return max(summaries, key=lambda path: path.stat().st_mtime)


def _find_coverage_summary(config: ServerConfig) -> tuple[Path | None, Path | None]:
    for root in _coverage_roots(config):
        if not root.exists():
            continue
        summary = _latest_summary_under(root)
        if summary is not None:
            return summary.resolve(), root.resolve()
    return None, None


def _coverage_error_detail(config: ServerConfig) -> str:
    if config.coverage_run is None:
        return (
            "Coverage is disabled because no --coverage_run folder was provided when "
            "launching the visualization server."
        )

    if not config.coverage_run.exists():
        return f"Coverage path does not exist: {config.coverage_run}"

    if config.coverage_run.is_file():
        return (
            "Coverage path points to a file, but the visualization expects a coverage run folder. "
            "Pass either a folder containing summary.json or a parent folder containing "
            "coverage_run_*/summary.json."
        )

    summary_path, _ = _find_coverage_summary(config)
    if summary_path is None:
        return (
            "Coverage folder format is not recognized. Expected summary.json directly inside "
            "the folder, or one or more coverage_run_* folders containing summary.json."
        )

    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except json.JSONDecodeError:
        return f"Coverage summary is not valid JSON: {summary_path}"

    if not isinstance(summary, dict):
        return f"Coverage summary has the wrong format: expected a JSON object in {summary_path}"

    if "images" not in summary or not isinstance(summary["images"], list):
        return f"Coverage summary is missing an images list: {summary_path}"

    return "Coverage summary could not be loaded."


def _cleanup_static_discoveries(config: ServerConfig) -> None:
    (config.static_dir / "discoveries.json").unlink(missing_ok=True)


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

        _compute_coordinates(
            config.discoveries,
            static_dir=config.static_dir,
            max_displayed=state.display_limit,
        )
        state.last_recompute_time = time.monotonic()
        return True


def _is_relevant_discovery_change(changes: set[tuple[Change, str]]) -> bool:
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
        if not _is_relevant_discovery_change(changes):
            continue

        print("Change in discoveries")
        time.sleep(RECOMPUTE_DEBOUNCE_SECONDS)
        if recompute_discoveries(config, state, respect_interval=True):
            print("Discoveries recomputed")
        else:
            print("Discovery recompute skipped: waiting for live-update interval")


def _coverage_image_url(image: str, run_dir: Path, serving_root: Path) -> str:
    image_path = (run_dir / image).resolve()
    if _is_relative_to(image_path, serving_root):
        return f"/coverage/{image_path.relative_to(serving_root).as_posix()}"
    return f"/coverage/{image}"


def _coverage_summary_payload(
    config: ServerConfig,
    summary_path: Path,
    serving_root: Path,
) -> dict[str, Any]:
    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=_coverage_error_detail(config))

    if not isinstance(summary, dict) or "images" not in summary or not isinstance(summary["images"], list):
        raise HTTPException(status_code=422, detail=_coverage_error_detail(config))

    run_dir = summary_path.parent
    summary["images"] = [
        {
            "file": image,
            "url": _coverage_image_url(image, run_dir, serving_root),
        }
        for image in summary.get("images", [])
    ]
    summary["run_name"] = run_dir.name
    return summary


def create_app(config: ServerConfig, state: RuntimeState | None = None) -> FastAPI:
    state = state or RuntimeState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.static_dir.mkdir(parents=True, exist_ok=True)
        config.discoveries.mkdir(parents=True, exist_ok=True)

        _compute_coordinates(
            config.discoveries,
            static_dir=config.static_dir,
            max_displayed=state.display_limit,
        )

        if config.refresh:
            thread = threading.Thread(
                target=watch_discoveries,
                args=(config, state),
                daemon=True,
            )
            thread.start()

        yield
        _cleanup_static_discoveries(config)

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount(
        "/static",
        StaticFiles(directory=config.static_dir, html=True),
        name="static",
    )

    @app.get("/discoveries/{file_path:path}")
    async def serve_discoveries(file_path: str):
        full_path = (config.discoveries / file_path).resolve()
        if not _is_relative_to(full_path, config.discoveries):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(full_path), media_type=_mime_type(file_path))

    @app.get("/coverage/{file_path:path}")
    async def serve_coverage_file(file_path: str):
        if config.coverage_run is None:
            raise HTTPException(status_code=404, detail="Coverage is disabled")

        full_path = None
        for root in _coverage_roots(config):
            candidate = (root / file_path).resolve()
            if _is_relative_to(candidate, root) and candidate.exists():
                full_path = candidate
                break

        if full_path is None:
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(str(full_path), media_type=_mime_type(file_path))

    @app.get("/coverage_status")
    async def coverage_status():
        return {
            "enabled": config.coverage_run is not None,
            "path": str(config.coverage_run) if config.coverage_run is not None else None,
        }

    @app.get("/coverage_summary")
    async def coverage_summary():
        if config.coverage_run is None:
            raise HTTPException(status_code=404, detail=_coverage_error_detail(config))
        if not config.coverage_run.exists():
            raise HTTPException(status_code=404, detail=_coverage_error_detail(config))
        if config.coverage_run.is_file():
            raise HTTPException(status_code=422, detail=_coverage_error_detail(config))

        summary_path, serving_root = _find_coverage_summary(config)
        if summary_path is None or serving_root is None:
            raise HTTPException(status_code=404, detail=_coverage_error_detail(config))

        return _coverage_summary_payload(config, summary_path, serving_root)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        print("Websocket connection")
        await websocket.accept()
        async for changes in awatch(config.static_dir):
            for change, path in changes:
                if change in (Change.added, Change.modified) and Path(path).name == "discoveries.json":
                    print("New coordinates file")
                    try:
                        await websocket.send_text("refresh")
                    except Exception:
                        return
                    break

    @app.get("/")
    async def read_root():
        return RedirectResponse(url="/static/index.html")

    @app.get("/display_limit")
    async def get_display_limit():
        return {
            "limit": state.display_limit,
            "default": DEFAULT_DISPLAY_LIMIT,
            "presets": DISPLAY_LIMIT_PRESETS,
            "min": MIN_DISPLAY_LIMIT,
            "max": MAX_DISPLAY_LIMIT,
        }

    @app.post("/display_limit")
    async def set_display_limit(payload: dict[str, Any]):
        try:
            limit = int(payload.get("limit"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Display limit must be an integer.")

        if limit < MIN_DISPLAY_LIMIT or limit > MAX_DISPLAY_LIMIT:
            raise HTTPException(
                status_code=422,
                detail=f"Display limit must be between {MIN_DISPLAY_LIMIT} and {MAX_DISPLAY_LIMIT}.",
            )

        state.display_limit = limit
        recompute_discoveries(config, state, ignore_interval=True)
        return {"status": "ok", "limit": state.display_limit}

    @app.post("/recompute_layout")
    async def recompute_layout():
        recompute_discoveries(config, state, ignore_interval=True)
        return {"status": "ok"}

    @app.post("/export")
    async def export_files(files: list[str]):
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        new_dir = (config.discoveries.parent / current_time).resolve()
        new_dir.mkdir(parents=True, exist_ok=True)

        copied_dirs = set()
        for file in files:
            normalized = file.lstrip("/")
            if normalized.startswith("discoveries/"):
                normalized = normalized[len("discoveries/"):]

            relative_dir = Path(normalized).parent
            if relative_dir == Path("."):
                continue

            source_dir = (config.discoveries / relative_dir).resolve()
            if not _is_relative_to(source_dir, config.discoveries):
                continue
            if not source_dir.exists() or not source_dir.is_dir():
                continue
            if source_dir in copied_dirs:
                continue

            copied_dirs.add(source_dir)
            destination = new_dir / source_dir.name
            shutil.copytree(source_dir, destination, dirs_exist_ok=True)

        return {"status": "ok", "new_dir": str(new_dir)}

    return app


def main() -> None:
    config = parse_args()
    app = create_app(config)

    import uvicorn

    try:
        uvicorn.run(app, host="127.0.0.1", port=8765)
    except KeyboardInterrupt:
        _cleanup_static_discoveries(config)


if __name__ == "__main__":
    main()
