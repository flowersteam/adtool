from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from watchfiles import Change, awatch

from adtool.examples.visu.analysis_jobs import random_run_payload, run_analysis_payload
from adtool.examples.visu.analysis_runs import (
    analysis_runs_dir,
    analysis_runs_payload,
    analysis_status_payload,
    latest_analysis_summary_payload,
)
from adtool.examples.visu.exporter import export_selected_discoveries
from adtool.examples.visu.highlights import materialize_discovery_filters
from adtool.examples.visu.layout import (
    cleanup_static_discoveries,
    recompute_discoveries,
    write_discovery_coordinates,
)
from adtool.examples.visu.online_update import (
    recompute_online_discoveries,
    start_online_updates,
)
from adtool.examples.visu.runtime import (
    DEFAULT_DISPLAY_LIMIT,
    ONLINE_FULL_RECOMPUTE_INTERVAL_SECONDS,
    ONLINE_POINT_UPDATE_INTERVAL_SECONDS,
    DEFAULT_PROJECTION_AXES,
    DEFAULT_PROJECTION_METHOD,
    DEFAULT_STICKER_PREVIEW_WORLD_HEIGHT,
    DISPLAY_LIMIT_PRESETS,
    MAX_DISPLAY_LIMIT,
    MIN_DISPLAY_LIMIT,
    MAX_STICKER_PREVIEW_WORLD_HEIGHT,
    MIN_STICKER_PREVIEW_WORLD_HEIGHT,
    PROJECTION_METHODS,
    RuntimeState,
    ServerConfig,
)
from adtool.examples.visu.server_support import is_relative_to, mime_type
from adtool.utils.interaction.experiment_control import (
    read_experiment_control,
    write_experiment_control,
)


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discoveries", type=str, required=True)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--refresh", action="store_true", help="Enable live online updates with incremental refresh and pause control.")
    args = parser.parse_args()

    return ServerConfig(
        discoveries=Path(args.discoveries).resolve(),
        config_file=Path(args.config_file).resolve() if args.config_file else None,
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


def _validate_projection_axes(raw_axes: Any) -> tuple[int, int]:
    if not isinstance(raw_axes, (list, tuple)) or len(raw_axes) != 2:
        raise HTTPException(status_code=422, detail="Projection axes must contain two ids.")

    try:
        x_axis = int(raw_axes[0])
        y_axis = int(raw_axes[1])
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Projection axes must be integers.")

    if x_axis < 0 or y_axis < 0:
        raise HTTPException(status_code=422, detail="Projection axes must be non-negative.")
    if x_axis == y_axis:
        raise HTTPException(status_code=422, detail="Projection axes must be different.")
    return x_axis, y_axis


def _validate_projection(
    payload: dict[str, Any],
    current_axes: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    method = str(payload.get("method", "")).strip().lower()
    if method not in PROJECTION_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"Projection method must be one of: {', '.join(PROJECTION_METHODS)}.",
        )

    axes = current_axes
    if "axes" in payload or method == "axis":
        axes = _validate_projection_axes(payload.get("axes", current_axes))

    return method, axes


def _validate_render_settings(payload: dict[str, Any]) -> float:
    try:
        sticker_preview_world_height = float(payload.get("sticker_preview_world_height"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Sticker preview size must be a number.")

    if (
        sticker_preview_world_height < MIN_STICKER_PREVIEW_WORLD_HEIGHT
        or sticker_preview_world_height > MAX_STICKER_PREVIEW_WORLD_HEIGHT
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Sticker preview size must be between "
                f"{MIN_STICKER_PREVIEW_WORLD_HEIGHT} and {MAX_STICKER_PREVIEW_WORLD_HEIGHT}."
            ),
        )

    return sticker_preview_world_height


def create_app(config: ServerConfig, state: RuntimeState | None = None) -> FastAPI:
    state = state or RuntimeState()

    def refresh_layout(ignore_interval: bool = True) -> None:
        if config.refresh:
            with state.recompute_lock:
                recompute_online_discoveries(config, state)
            return
        recompute_discoveries(config, state, ignore_interval=ignore_interval)

    def runtime_status_payload() -> dict[str, Any]:
        control = read_experiment_control(config.discoveries)
        if config.refresh:
            mode = "refresh"
        else:
            mode = "manual"

        return {
            "mode": mode,
            "refresh": config.refresh,
            "pause_supported": config.refresh,
            "paused": control["paused"],
            "point_update_interval_seconds": ONLINE_POINT_UPDATE_INTERVAL_SECONDS,
            "full_recompute_interval_seconds": ONLINE_FULL_RECOMPUTE_INTERVAL_SECONDS,
        }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.static_dir.mkdir(parents=True, exist_ok=True)
        config.discoveries.mkdir(parents=True, exist_ok=True)
        analysis_runs_dir(config).mkdir(parents=True, exist_ok=True)
        if config.refresh:
            start_online_updates(config, state)
        else:
            write_discovery_coordinates(config, state)

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

    @app.get("/analysis_files/{file_path:path}")
    async def serve_analysis_file(file_path: str):
        runs_dir = analysis_runs_dir(config)
        full_path = (runs_dir / file_path).resolve()
        if not is_relative_to(full_path, runs_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(str(full_path), media_type=mime_type(file_path))

    @app.get("/analysis_status")
    async def analysis_status():
        return analysis_status_payload(config)

    @app.get("/analysis_summary")
    async def analysis_summary():
        return latest_analysis_summary_payload(config)

    @app.get("/analysis_runs")
    async def analysis_runs():
        return analysis_runs_payload(config)

    @app.get("/runtime_status")
    async def runtime_status():
        return runtime_status_payload()

    @app.get("/experiment_control")
    async def experiment_control():
        return read_experiment_control(config.discoveries)

    @app.post("/experiment_control")
    async def set_experiment_control(payload: dict[str, Any]):
        if not config.refresh:
            raise HTTPException(status_code=400, detail="Experiment control is only available when live refresh is enabled.")
        paused = bool(payload.get("paused", False))
        return write_experiment_control(config.discoveries, paused)

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
        old_limit = state.display_limit
        state.display_limit = _validate_display_limit(payload)
        try:
            refresh_layout(ignore_interval=True)
        except ValueError as error:
            state.display_limit = old_limit
            raise HTTPException(status_code=422, detail=str(error))
        return {"status": "ok", "limit": state.display_limit}

    @app.get("/projection")
    async def get_projection():
        return {
            "method": state.projection_method,
            "axes": list(state.projection_axes),
            "default_method": DEFAULT_PROJECTION_METHOD,
            "default_axes": list(DEFAULT_PROJECTION_AXES),
            "methods": PROJECTION_METHODS,
        }

    @app.post("/projection")
    async def set_projection(payload: dict[str, Any]):
        method, axes = _validate_projection(payload, state.projection_axes)
        old_method = state.projection_method
        old_axes = state.projection_axes
        state.projection_method = method
        state.projection_axes = axes
        try:
            refresh_layout(ignore_interval=True)
        except ValueError as error:
            state.projection_method = old_method
            state.projection_axes = old_axes
            raise HTTPException(status_code=422, detail=str(error))
        return {
            "status": "ok",
            "method": state.projection_method,
            "axes": list(state.projection_axes),
        }

    @app.get("/render_settings")
    async def get_render_settings():
        return {
            "sticker_preview_world_height": state.sticker_preview_world_height,
            "default_sticker_preview_world_height": DEFAULT_STICKER_PREVIEW_WORLD_HEIGHT,
            "min_sticker_preview_world_height": MIN_STICKER_PREVIEW_WORLD_HEIGHT,
            "max_sticker_preview_world_height": MAX_STICKER_PREVIEW_WORLD_HEIGHT,
        }

    @app.post("/render_settings")
    async def set_render_settings(payload: dict[str, Any]):
        sticker_preview_world_height = _validate_render_settings(payload)
        state.sticker_preview_world_height = sticker_preview_world_height
        return {
            "status": "ok",
            "sticker_preview_world_height": state.sticker_preview_world_height,
        }

    @app.post("/recompute_layout")
    async def recompute_layout():
        try:
            refresh_layout(ignore_interval=True)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        return {"status": "ok"}

    @app.post("/discovery_highlights/materialize")
    async def materialize_highlight_filters():
        try:
            result = materialize_discovery_filters(
                config.discoveries,
                config_path=config.config_file,
            )
            refresh_layout(ignore_interval=True)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        return {"status": "ok", **result}

    @app.post("/analysis/random_run")
    def random_run(payload: dict[str, Any]):
        return random_run_payload(config, state, payload)

    @app.post("/analysis/run")
    def run_analysis(payload: dict[str, Any]):
        return run_analysis_payload(config, state, payload)

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
