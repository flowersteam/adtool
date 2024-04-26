#!/usr/bin/env python3
import os
import asyncio
import datetime
import functools
import websockets
from http import HTTPStatus
#watch discovery_files for a change 
from coordinates import compute_coordinates
from watchfiles import awatch, Change


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


async def process_request( path, request_headers):
    """Serves a file when doing a GET request with a valid path."""

    if "Upgrade" in request_headers:
        return  # Probably a WebSocket connection

    if path == '/':
        path = '/index.html'

    response_headers = [
        ('Server', 'asyncio websocket server'),
        ('Connection', 'close'),
    ]

    print("HTTP GET", path)


    if path.startswith("/discoveries/"):
        path=os.path.join(discovery_files, path[ len("/discoveries/") :])
    else:
        path=os.path.join(static_files, path[1:])

    # Guess file content type
    extension = path.split(".")[-1]
    mime_type = MIME_TYPES.get(extension, "application/octet-stream")
    response_headers.append(('Content-Type', mime_type))

    # Read the whole file into memory and send it out
    #if not exist return 404
    
    if not os.path.exists(path):
        return HTTPStatus.NOT_FOUND, response_headers, b"404 Not Found"
    body = open(path, 'rb').read()
    response_headers.append(('Content-Length', str(len(body))))
    #print("HTTP GET {} 200 OK".format(path))
    return HTTPStatus.OK, response_headers, body


async def watch_discoveries():
    async for changes in awatch(discovery_files, recursive=True):
        for _ in changes:
            
            print("Change in discoveries")
            compute_coordinates(discovery_files)
            break




async def watch_coordinates(websocket):
    while True:
        async for changes in awatch(f"{static_files}/discoveries.json"):
            for change in changes:
                if change[0] in  (  Change.added  , Change.modified) :
                    print("New coordinates file")
                    try:
                        await websocket.send("refresh")
                        break
                    except (websockets.exceptions.ConnectionClosedError,websockets.exceptions.ConnectionClosedOK):
                        return
                

compute_coordinates( discovery_files   )    




if __name__ == "__main__":
    try:
        # set first argument for the handler to current working directory
        handler = functools.partial(process_request)
        start_server = websockets.serve(watch_coordinates, '127.0.0.1', 8765,
                                        process_request=handler)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_server)
        #start watching for new discoveries
        server=loop.create_task(watch_discoveries())
        print("Server has started at http://127.0.0.1:8765")
     

        loop.run_forever()
    except KeyboardInterrupt:
        # delete static/discoveires.json
        if os.path.exists(f"{static_files}/discoveries.json"):
            os.remove(f"{static_files}/discoveries.json")

        os._exit(0)