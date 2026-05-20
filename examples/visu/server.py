import argparse
import os
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from coordinates import compute_coordinates
from watchfiles import awatch, Change, watch
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import threading
import shutil
from typing import Optional

from datetime import datetime

MIME_TYPES = {
    "html": "text/html",
    "js": "text/javascript",
    "css": "text/css",
    "mjs": "text/javascript",
    "json": "application/json",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


BASE_DIR = Path(__file__).resolve().parent
static_files = BASE_DIR / "static"


parser = argparse.ArgumentParser()
parser.add_argument("--discoveries", type=str, required=True)
parser.add_argument("--coverage_run", type=str, required=False, default=None)
args = parser.parse_args()

discovery_files = Path(args.discoveries).resolve()
coverage_run = (
    Path(args.coverage_run).resolve()
    if args.coverage_run
    else (discovery_files.parent / "coverage_run").resolve()
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _coverage_roots() -> list[Path]:
    roots = [
        coverage_run,
        discovery_files.parent / "coverage_runs",
        discovery_files.parent / "coverage_run",
    ]

    if coverage_run.name.startswith("coverage_run_"):
        roots.append(coverage_run.parent)

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def _latest_summary_under(root: Path) -> Optional[Path]:
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


def _find_coverage_summary() -> tuple[Optional[Path], Optional[Path]]:
    for root in _coverage_roots():
        if not root.exists():
            continue
        summary = _latest_summary_under(root)
        if summary is not None:
            return summary.resolve(), root.resolve()
    return None, None


def recompute_discoveries() -> None:
    """Recompute static coordinate artifacts from current discoveries."""
    compute_coordinates(
        str(discovery_files),
        static_dir=str(static_files),
    )


def watch_discoveries():
    print("Watching discoveries")
    for changes in watch(str(discovery_files), recursive=True):
        # Only recompute for meaningful data updates.
        has_relevant_change = any(
            change[0] in (Change.added, Change.modified, Change.deleted)
            and Path(change[1]).name in {"discovery.json", "visu.png", "config.json"}
            for change in changes
        )
        if not has_relevant_change:
            continue

        print("Change in discoveries")
        recompute_discoveries()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure directories exist
    os.makedirs(static_files, exist_ok=True)
    os.makedirs(discovery_files, exist_ok=True)

    recompute_discoveries()

    # Keep live refresh on by default to support concurrent experimentation.
    t = threading.Thread(target=watch_discoveries, daemon=True)
    t.start()

    yield
    # delete static/discoveires.json
    discoveries_json = static_files / "discoveries.json"
    if discoveries_json.exists():
        discoveries_json.unlink()

    os._exit(0)


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory=str(static_files),
          html=True), name="static")


@app.get("/discoveries/{file_path:path}")
async def serve_discoveries(file_path: str):
    full_path = (discovery_files / file_path).resolve()
    print(full_path)

    if not str(full_path).startswith(str(discovery_files)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    extension = file_path.split(".")[-1]
    mime_type = MIME_TYPES.get(extension, "application/octet-stream")

    return FileResponse(str(full_path), media_type=mime_type)


@app.get("/coverage/{file_path:path}")
async def serve_coverage_file(file_path: str):
    full_path = None
    for root in _coverage_roots():
        candidate = (root / file_path).resolve()
        if _is_relative_to(candidate, root) and candidate.exists():
            full_path = candidate
            break

    if full_path is None:
        raise HTTPException(status_code=404, detail="File not found")

    extension = file_path.split(".")[-1]
    mime_type = MIME_TYPES.get(extension, "application/octet-stream")
    return FileResponse(str(full_path), media_type=mime_type)


@app.get("/coverage_summary")
async def coverage_summary():
    summary_path, serving_root = _find_coverage_summary()
    if summary_path is None or serving_root is None:
        raise HTTPException(status_code=404, detail="Coverage summary not found")

    with open(summary_path) as handle:
        summary = json.load(handle)

    run_dir = summary_path.parent
    images = []
    for image in summary.get("images", []):
        image_path = (run_dir / image).resolve()
        if _is_relative_to(image_path, serving_root):
            image_url = f"/coverage/{image_path.relative_to(serving_root).as_posix()}"
        else:
            image_url = f"/coverage/{image}"

        images.append({
            "file": image,
            "url": image_url,
        })

    summary["images"] = images
    summary["run_name"] = run_dir.name
    return summary


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Websocket connection")
    await websocket.accept()
    while True:
        async for changes in awatch(str(static_files)):
            for change in changes:
                if change[0] in (Change.added, Change.modified):
                    print("New coordinates file")
                    try:
                        await websocket.send_text("refresh")
                        break
                    except Exception:
                        return


@app.get("/")
async def read_root():
    return RedirectResponse(url="/static/index.html")


# curl 'http://127.0.0.1:8765/export' -X POST -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0' -H 'Accept: */*' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'Referer: http://127.0.0.1:8765/static/index.html' -H 'Content-Type: application/json' -H 'Origin: http://127.0.0.1:8765' -H 'Connection: keep-alive' -H 'Sec-Fetch-Dest: empty' -H 'Sec-Fetch-Mode: cors' -H 'Sec-Fetch-Site: same-origin' -H 'Priority: u=1' -H 'Pragma: no-cache' -H 'Cache-Control: no-cache' --data-raw '["/2024-06-06T15:16_exp_0_idx_295_seed_42/4e50156e34f55df28f88bf68a82688e58115f5a5.mp4","/2024-06-06T15:16_exp_0_idx_296_seed_42/11ab8da6166086f334b23b15bd70a27b3d76e54a.mp4","/2024-06-06T15:17_exp_0_idx_297_seed_42/0b1d886f90161aaadd5bdd4018d0486817a02091.mp4"]'

@app.post("/export")
async def export_files(files: list[str]):
    print(files)
    # create a new directory with the date
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    new_dir = (discovery_files.parent / current_time).resolve()

    os.makedirs(new_dir, exist_ok=True)

    copied_dirs = set()
    for file in files:
        normalized = file.lstrip("/")
        if normalized.startswith("discoveries/"):
            normalized = normalized[len("discoveries/"):]

        relative_dir = Path(normalized).parent
        if relative_dir == Path("."):
            continue

        source_dir = (discovery_files / relative_dir).resolve()
        if not str(source_dir).startswith(str(discovery_files)):
            continue
        if not source_dir.exists() or not source_dir.is_dir():
            continue

        if source_dir in copied_dirs:
            continue
        copied_dirs.add(source_dir)

        destination = new_dir / source_dir.name
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)

    return {"status": "ok", "new_dir": str(new_dir)}


if __name__ == "__main__":
    # start_server = websockets.serve(websocket_endpoint, '127.0.0.1', 8766,
    #                                 process_request= app)
    # asyncio.get_event_loop().run_until_complete(start_server)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8765)
    except KeyboardInterrupt:
        # delete static/discoveires.json
        discoveries_json = static_files / "discoveries.json"
        if discoveries_json.exists():
            discoveries_json.unlink()

        os._exit(0)
