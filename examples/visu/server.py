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

from datetime import datetime

current_pca = None

MIME_TYPES = {
    "html": "text/html",
    "js": "text/javascript",
    "css": "text/css",
    "mjs": "text/javascript",
    "json": "application/json",
}


BASE_DIR = Path(__file__).resolve().parent
static_files = BASE_DIR / "static"


parser = argparse.ArgumentParser()
parser.add_argument("--discoveries", type=str, required=True)
args = parser.parse_args()

discovery_files = Path(args.discoveries).resolve()


def recompute_discoveries() -> None:
    """Recompute static coordinate artifacts from current discoveries."""
    global current_pca
    current_pca = compute_coordinates(
        str(discovery_files),
        static_dir=str(static_files),
    )


def watch_discoveries():
    print("Watching discoveries")
    for changes in watch(str(discovery_files), recursive=True):
        if any(Path(change[1]).name == "target.json" for change in changes):
            continue

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


@app.get("/disable_target")
async def disable_target():
    # delete static/target.json
    target_file = discovery_files / "target.json"
    if target_file.exists():
        target_file.unlink()
    return {"status": "ok"}


@app.post("/update_target")
async def update_target(target: dict):
    if current_pca is None:
        raise HTTPException(status_code=503, detail="No coordinates available")

    # transform x and y to float and get the reverse pca
    x, y = target['x'], target['y']
    x, y = float(x), float(y)
    target_embedding = current_pca.inverse_transform([[x, y]])
    target['target'] = target_embedding[0].tolist()
    with open(discovery_files / "target.json", "w") as f:
        json.dump(target, f)

    return {"status": "ok"}


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
