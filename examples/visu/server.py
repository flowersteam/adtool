from __future__ import annotations

import argparse
import json
import shutil
import sys
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
    "svg": "image/svg+xml",
    "mp4": "video/mp4",
    "webm": "video/webm",
}

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
STATIC_DIR = BASE_DIR / "static"
RECOMPUTE_DEBOUNCE_SECONDS = 10.0
RECOMPUTE_MIN_INTERVAL_SECONDS = 15.0
DEFAULT_DISPLAY_LIMIT = 500
MIN_DISPLAY_LIMIT = 1
MAX_DISPLAY_LIMIT = 10000
DISPLAY_LIMIT_PRESETS = [250, 500, 1000, 1500, 2000]
DEFAULT_RANDOM_ITERATIONS = 100
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class ServerConfig:
    discoveries: Path
    static_dir: Path = STATIC_DIR
    refresh: bool = False


@dataclass
class RuntimeState:
    display_limit: int = DEFAULT_DISPLAY_LIMIT
    last_recompute_time: float = 0.0
    analysis_lock: threading.Lock = field(default_factory=threading.Lock)
    recompute_lock: threading.Lock = field(default_factory=threading.Lock)


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discoveries", type=str, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    return ServerConfig(
        discoveries=Path(args.discoveries).resolve(),
        refresh=args.refresh,
    )


def _compute_coordinates(*args, **kwargs) -> None:
    try:
        from .coordinates import compute_coordinates
    except ImportError:
        from coordinates import compute_coordinates

    compute_coordinates(*args, **kwargs)


def _ensure_analysis_import_paths() -> None:
    for import_root in (REPO_ROOT, EXAMPLES_DIR):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))


def _load_random_baseline_runner():
    _ensure_analysis_import_paths()
    try:
        from analysis_metrics.random_run import run_random_baseline
    except ModuleNotFoundError as exc:
        if exc.name != "analysis_metrics":
            raise
        from examples.analysis_metrics.random_run import run_random_baseline

    return run_random_baseline


def _load_coverage_comparator():
    _ensure_analysis_import_paths()
    try:
        from analysis_metrics.coverage_comparison import compare_discovery_sets
    except ModuleNotFoundError as exc:
        if exc.name != "analysis_metrics":
            raise
        from examples.analysis_metrics.coverage_comparison import compare_discovery_sets

    return compare_discovery_sets


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _mime_type(path: str | Path) -> str:
    extension = str(path).rsplit(".", maxsplit=1)[-1].lower()
    return MIME_TYPES.get(extension, "application/octet-stream")


def _coverage_runs_dir(config: ServerConfig) -> Path:
    return (config.discoveries.parent / "coverage_runs").resolve()


def _coverage_summary_paths(coverage_runs_dir: Path) -> list[Path]:
    if not coverage_runs_dir.exists() or not coverage_runs_dir.is_dir():
        return []

    summaries = [
        summary
        for summary in coverage_runs_dir.glob("coverage_run_*/summary.json")
        if summary.is_file()
    ]
    return sorted(summaries, key=lambda path: path.stat().st_mtime, reverse=True)


def _coverage_summary_error(summary_path: Path) -> str:
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
    summary_path: Path,
    serving_root: Path,
) -> dict[str, Any]:
    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=_coverage_summary_error(summary_path))

    if not isinstance(summary, dict) or "images" not in summary or not isinstance(summary["images"], list):
        raise HTTPException(status_code=422, detail=_coverage_summary_error(summary_path))

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


def _resolve_input_path(value: Any, field_name: str, required: bool = True) -> Path | None:
    if value is None or str(value).strip() == "":
        if required:
            raise HTTPException(status_code=422, detail=f"{field_name} is required.")
        return None

    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()

    bases = (Path.cwd(), REPO_ROOT, EXAMPLES_DIR)
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate

    return (Path.cwd() / path).resolve()


def _require_directory(path: Path, field_name: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{field_name} does not exist: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=422, detail=f"{field_name} must be a directory: {path}")


def _require_file(path: Path, field_name: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise HTTPException(status_code=422, detail=f"{field_name} must be a file: {path}")


def _payload_int(
    payload: dict[str, Any],
    field_name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = payload.get(field_name, default)
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{field_name} must be an integer.")

    if minimum is not None and parsed < minimum:
        raise HTTPException(status_code=422, detail=f"{field_name} must be at least {minimum}.")
    if maximum is not None and parsed > maximum:
        raise HTTPException(status_code=422, detail=f"{field_name} must be at most {maximum}.")
    return parsed


def _optional_payload_int(payload: dict[str, Any], field_name: str) -> int | None:
    if field_name not in payload or payload.get(field_name) in (None, ""):
        return None
    return _payload_int(payload, field_name, 0, minimum=2)


def _timestamped_analysis_dir(config: ServerConfig, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (config.discoveries.parent / "analysis_runs" / f"{prefix}_{timestamp}").resolve()


def _error_detail(prefix: str, exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    return f"{prefix}: {message}"


def create_app(config: ServerConfig, state: RuntimeState | None = None) -> FastAPI:
    state = state or RuntimeState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.static_dir.mkdir(parents=True, exist_ok=True)
        config.discoveries.mkdir(parents=True, exist_ok=True)
        _coverage_runs_dir(config).mkdir(parents=True, exist_ok=True)

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
        coverage_runs_dir = _coverage_runs_dir(config)
        full_path = (coverage_runs_dir / file_path).resolve()
        if not _is_relative_to(full_path, coverage_runs_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(str(full_path), media_type=_mime_type(file_path))

    @app.get("/coverage_status")
    async def coverage_status():
        coverage_runs_dir = _coverage_runs_dir(config)
        summaries = _coverage_summary_paths(coverage_runs_dir)
        return {
            "enabled": True,
            "has_run": len(summaries) > 0,
            "path": str(coverage_runs_dir),
            "run_count": len(summaries),
        }

    @app.get("/coverage_summary")
    async def coverage_summary():
        coverage_runs_dir = _coverage_runs_dir(config)
        summaries = _coverage_summary_paths(coverage_runs_dir)
        if not summaries:
            raise HTTPException(
                status_code=404,
                detail=f"No coverage runs found in: {coverage_runs_dir}",
            )

        return _coverage_summary_payload(summaries[0], coverage_runs_dir)

    @app.get("/coverage_runs")
    async def coverage_runs():
        coverage_runs_dir = _coverage_runs_dir(config)
        runs = [
            _coverage_summary_payload(summary_path, coverage_runs_dir)
            for summary_path in _coverage_summary_paths(coverage_runs_dir)
        ]
        return {
            "coverage_runs_dir": str(coverage_runs_dir),
            "runs": runs,
        }

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

    @app.post("/analysis/random_run")
    def random_run(payload: dict[str, Any]):
        config_file = _resolve_input_path(payload.get("config_file"), "config_file")
        if config_file is None:
            raise HTTPException(status_code=422, detail="config_file is required.")
        _require_file(config_file, "config_file")

        output_dir = _resolve_input_path(payload.get("output_dir"), "output_dir", required=False)
        if output_dir is None:
            output_dir = _timestamped_analysis_dir(config, "random_run")

        nb_iterations = _payload_int(
            payload,
            "nb_iterations",
            DEFAULT_RANDOM_ITERATIONS,
            minimum=1,
        )
        seed = _payload_int(
            payload,
            "seed",
            DEFAULT_RANDOM_SEED,
            minimum=0,
            maximum=2**32 - 1,
        )

        runner = _load_random_baseline_runner()
        with state.analysis_lock:
            try:
                summary = runner(
                    config_file=config_file,
                    output_dir=output_dir,
                    nb_iterations=nb_iterations,
                    seed=seed,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=_error_detail("Random run failed", exc),
                ) from exc

        return {
            "status": "ok",
            "output_dir": str(summary.output_dir),
            "discoveries_dir": str(summary.discoveries_dir),
            "count": summary.count,
            "seed": summary.seed,
        }

    @app.post("/analysis/coverage_comparison")
    def coverage_comparison(payload: dict[str, Any]):
        comparison_path = _resolve_input_path(
            payload.get("path", payload.get("comparison_path")),
            "path",
        )
        if comparison_path is None:
            raise HTTPException(status_code=422, detail="path is required.")
        _require_directory(comparison_path, "path")

        raw_config_file = payload.get("config_file")
        if isinstance(raw_config_file, str) and raw_config_file.strip().lower() == "none":
            raw_config_file = None
        config_file = _resolve_input_path(raw_config_file, "config_file", required=False)
        if config_file is not None:
            _require_file(config_file, "config_file")

        points = _optional_payload_int(payload, "points")
        label_a = payload.get("label_a") or "IMGEP"
        label_b = payload.get("label_b") or "baseline"

        comparator = _load_coverage_comparator()
        with state.analysis_lock:
            try:
                summary = comparator(
                    config.discoveries,
                    comparison_path,
                    output_dir=_coverage_runs_dir(config),
                    label_a=str(label_a),
                    label_b=str(label_b),
                    config_file=config_file,
                    points=points,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=_error_detail("Coverage comparison failed", exc),
                ) from exc

        return {
            "status": "ok",
            "run_dir": str(summary.run_dir),
            "dataset_a_path": str(summary.discovery_a_path),
            "dataset_b_path": str(summary.discovery_b_path),
            "dataset_a_count": summary.count_a,
            "dataset_b_count": summary.count_b,
            "dim_count": summary.dim_count,
            "images": summary.images,
        }

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
