from __future__ import annotations

import argparse
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from watchfiles import Change, awatch

if __package__:
    from .analysis_jobs import coverage_comparison_payload, random_run_payload
    from .coverage_runs import (
        coverage_runs_dir,
        coverage_runs_payload,
        coverage_status_payload,
        latest_coverage_summary_payload,
    )
    from .exporter import export_selected_discoveries
    from .layout import (
        cleanup_static_discoveries,
        recompute_discoveries,
        watch_discoveries,
        write_discovery_coordinates,
    )
    from .runtime import (
        DEFAULT_DISPLAY_LIMIT,
        DISPLAY_LIMIT_PRESETS,
        MAX_DISPLAY_LIMIT,
        MIN_DISPLAY_LIMIT,
        RuntimeState,
        ServerConfig,
    )
    from .server_support import is_relative_to, mime_type
else:
    from analysis_jobs import coverage_comparison_payload, random_run_payload
    from coverage_runs import (
        coverage_runs_dir,
        coverage_runs_payload,
        coverage_status_payload,
        latest_coverage_summary_payload,
    )
    from exporter import export_selected_discoveries
    from layout import (
        cleanup_static_discoveries,
        recompute_discoveries,
        watch_discoveries,
        write_discovery_coordinates,
    )
    from runtime import (
        DEFAULT_DISPLAY_LIMIT,
        DISPLAY_LIMIT_PRESETS,
        MAX_DISPLAY_LIMIT,
        MIN_DISPLAY_LIMIT,
        RuntimeState,
        ServerConfig,
    )
    from server_support import is_relative_to, mime_type


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discoveries", type=str, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    return ServerConfig(
        discoveries=Path(args.discoveries).resolve(),
        refresh=args.refresh,
    )


def _validate_display_limit(payload: dict[str, Any]) -> int:
    try:
        limit = int(payload.get("limit"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Display limit must be an integer.")

    if limit < MIN_DISPLAY_LIMIT or limit > MAX_DISPLAY_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=f"Display limit must be between {MIN_DISPLAY_LIMIT} and {MAX_DISPLAY_LIMIT}.",
        )
    return limit


def create_app(config: ServerConfig, state: RuntimeState | None = None) -> FastAPI:
    state = state or RuntimeState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.static_dir.mkdir(parents=True, exist_ok=True)
        config.discoveries.mkdir(parents=True, exist_ok=True)
        coverage_runs_dir(config).mkdir(parents=True, exist_ok=True)
        write_discovery_coordinates(config, state)

        if config.refresh:
            thread = threading.Thread(
                target=watch_discoveries,
                args=(config, state),
                daemon=True,
            )
            thread.start()

        yield
        cleanup_static_discoveries(config)

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
        if not is_relative_to(full_path, config.discoveries):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(full_path), media_type=mime_type(file_path))

    @app.get("/coverage/{file_path:path}")
    async def serve_coverage_file(file_path: str):
        runs_dir = coverage_runs_dir(config)
        full_path = (runs_dir / file_path).resolve()
        if not is_relative_to(full_path, runs_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(str(full_path), media_type=mime_type(file_path))

    @app.get("/coverage_status")
    async def coverage_status():
        return coverage_status_payload(config)

    @app.get("/coverage_summary")
    async def coverage_summary():
        return latest_coverage_summary_payload(config)

    @app.get("/coverage_runs")
    async def coverage_runs():
        return coverage_runs_payload(config)

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
        state.display_limit = _validate_display_limit(payload)
        recompute_discoveries(config, state, ignore_interval=True)
        return {"status": "ok", "limit": state.display_limit}

    @app.post("/recompute_layout")
    async def recompute_layout():
        recompute_discoveries(config, state, ignore_interval=True)
        return {"status": "ok"}

    @app.post("/analysis/random_run")
    def random_run(payload: dict[str, Any]):
        return random_run_payload(config, state, payload)

    @app.post("/analysis/coverage_comparison")
    def coverage_comparison(payload: dict[str, Any]):
        return coverage_comparison_payload(config, state, payload)

    @app.post("/export")
    async def export_files(files: list[str]):
        return export_selected_discoveries(config, files)

    return app


def main() -> None:
    config = parse_args()
    app = create_app(config)

    import uvicorn

    try:
        uvicorn.run(app, host="127.0.0.1", port=8765)
    except KeyboardInterrupt:
        cleanup_static_discoveries(config)


if __name__ == "__main__":
    main()
