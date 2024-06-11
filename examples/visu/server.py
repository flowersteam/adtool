import os
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from coordinates import compute_coordinates
from watchfiles import awatch, Change
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import websockets
from pathlib import Path

from datetime import datetime


MIME_TYPES = {
    "html": "text/html",
    "js": "text/javascript",
    "css": "text/css",
    "mjs": "text/javascript",
    "json": "application/json",
}


static_files="static"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--discoveries", type=str, required=True)
args = parser.parse_args()

discovery_files=args.discoveries


async def watch_discoveries():
    print("Watching discoveries")
    async for changes in awatch(discovery_files, recursive=True):
        for _ in changes:
            print("Change in discoveries")
            compute_coordinates(discovery_files)
            break



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure directories exist
    os.makedirs(static_files, exist_ok=True)
    os.makedirs(discovery_files, exist_ok=True)
    
    compute_coordinates(discovery_files)
    task = asyncio.create_task(watch_discoveries())
    yield
    task.cancel()
    await task

    if os.path.exists(f"{static_files}/discoveries.json"):
        os.remove(f"{static_files}/discoveries.json")



app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory=static_files,html = True), name="static")




@app.get("/discoveries/{file_path:path}")
async def serve_discoveries(file_path: str):
    full_path = os.path.join(discovery_files, file_path)
    
    print(full_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    extension = file_path.split(".")[-1]
    mime_type = MIME_TYPES.get(extension, "application/octet-stream")

    return FileResponse(full_path, media_type=mime_type)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Websocket connection")
    await websocket.accept()
    while True:
        async for changes in awatch(f"{static_files}/discoveries.json"):
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


#curl 'http://127.0.0.1:8765/export' -X POST -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0' -H 'Accept: */*' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'Referer: http://127.0.0.1:8765/static/index.html' -H 'Content-Type: application/json' -H 'Origin: http://127.0.0.1:8765' -H 'Connection: keep-alive' -H 'Sec-Fetch-Dest: empty' -H 'Sec-Fetch-Mode: cors' -H 'Sec-Fetch-Site: same-origin' -H 'Priority: u=1' -H 'Pragma: no-cache' -H 'Cache-Control: no-cache' --data-raw '["/2024-06-06T15:16_exp_0_idx_295_seed_42/4e50156e34f55df28f88bf68a82688e58115f5a5.mp4","/2024-06-06T15:16_exp_0_idx_296_seed_42/11ab8da6166086f334b23b15bd70a27b3d76e54a.mp4","/2024-06-06T15:17_exp_0_idx_297_seed_42/0b1d886f90161aaadd5bdd4018d0486817a02091.mp4"]'

@app.post("/export")
async def export_files(files: list[str]):
    print(files)
    #create a new directory with the date 
    current_time= datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    new_dir = f"{discovery_files}/../{current_time}"

    # simplify path
    new_dir = os.path.abspath(new_dir)

    os.makedirs(new_dir, exist_ok=True)
    #copy past all files to a new directory
    for file in files:
        file_path=Path(file)
        file_path =  "/".join(file_path.parts[:-1])
        file_path = f"{discovery_files}/{file_path}"   
        
        os.system(f"cp -r {file_path} {new_dir}")
    return {"status": "ok", "new_dir": new_dir}
    

if __name__ == "__main__":
    # start_server = websockets.serve(websocket_endpoint, '127.0.0.1', 8766,
    #                                 process_request= app)
    # asyncio.get_event_loop().run_until_complete(start_server)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8765)
    except KeyboardInterrupt:
        # delete static/discoveires.json
        if os.path.exists(f"{static_files}/discoveries.json"):
            os.remove(f"{static_files}/discoveries.json")

        os._exit(0)
