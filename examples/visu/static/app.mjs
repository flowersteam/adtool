import * as PIXI from 'https://unpkg.com/pixi.js@8.x/dist/pixi.min.mjs';

let coordinates = [];
let zoomLevel = 1;
let sprites = [];
let draggedCoordinates = { x: 0, y: 0 };
const ZOOM_FACTOR = 100;

function computeCoordinates() {
    for (let i = 0; i < sprites.length; i++) {
        sprites[i].position.set(
            app.renderer.screen.width / 2 +
            zoomLevel * (ZOOM_FACTOR * coordinates[i].x + draggedCoordinates.x) + draggedCoordinates.x,
            app.renderer.screen.height / 2 + zoomLevel * (ZOOM_FACTOR * coordinates[i].y + draggedCoordinates.y) + draggedCoordinates.y
        );
    }
}

window.addEventListener('wheel', function (event) {
    zoomLevel += (-event.deltaY / ZOOM_FACTOR) * zoomLevel ** 0.5;
    zoomLevel = Math.max(1, zoomLevel);
    computeCoordinates();
});

let isDragging = false;
let lastMousePosition = { x: 0, y: 0 };
window.addEventListener('mousedown', function (event) {
    isDragging = true;
    lastMousePosition = { x: event.clientX, y: event.clientY };
});
window.addEventListener('mouseup', function (event) {
    isDragging = false;
});
window.addEventListener('mousemove', function (event) {
    if (isDragging) {
        draggedCoordinates.x += (event.clientX - lastMousePosition.x) / zoomLevel;
        draggedCoordinates.y += (event.clientY - lastMousePosition.y) / zoomLevel;
        lastMousePosition = { x: event.clientX, y: event.clientY };
        computeCoordinates();
    }
});

const app = new PIXI.Application({
    backgroundColor: 0xefffff,
    resizeTo: window
});

document.getElementById('canvas-container').appendChild(app.view);

let overlayText = document.getElementById('overlay-text');
let selectedSprites = [];

async function loadAssets() {
    coordinates = await PIXI.Assets.load('/discoveries.json');
    let concatenatedVisuals = await PIXI.Assets.load('concatenated.webm');

    const numVideos = coordinates.length;
    const rows = Math.ceil(Math.sqrt(numVideos));
    const cols = Math.ceil(numVideos / rows);

    for (let i = 0; i < coordinates.length; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;

        const shift_x = col * coordinates[i].width;
        const shift_y = row * coordinates[i].height;

        let subTexture = new PIXI.Texture({
            source: concatenatedVisuals.source,
            frame: new PIXI.Rectangle(shift_x, shift_y, coordinates[i].width, coordinates[i].height),
        });

        subTexture.baseTexture.resource.loop = true;
        let sprite = new PIXI.Sprite(subTexture);

        sprite.interactive = true;
        sprite.on('mouseover', (function (i) {
            return function () {
                overlayText.innerHTML = coordinates[i].visual;
            };
        })(i));

        sprite.on('click', (function (i) {
            return function () {
                addSelectedDiscovery(coordinates[i].visual);
            };
        })(i));

        sprite.anchor.set(0.5);
        sprite.x = shift_x + coordinates[i].width / 2;
        sprite.y = shift_y + coordinates[i].height / 2;

        sprites.push(sprite);
        app.stage.addChild(sprite);
    }

    computeCoordinates();
}

function addSelectedDiscovery(path) {
    if (!selectedSprites.includes(path)) {
        selectedSprites.push(path);
        updateSelectedList();
    }
}

function removeSelectedDiscovery(path) {
    const index = selectedSprites.indexOf(path);
    if (index > -1) {
        selectedSprites.splice(index, 1);
        updateSelectedList();
    }
}

function updateSelectedList() {
    const selectedListContainer = document.getElementById('selected-sprites');
    selectedListContainer.innerHTML = '';
    selectedSprites.forEach(path => {
        const div = document.createElement('div');
        div.className = 'sprite-item';
        div.innerHTML = path;
        const removeButton = document.createElement('button');
        removeButton.className = 'remove-button';
        removeButton.innerHTML = 'Remove';
        removeButton.onclick = () => removeSelectedDiscovery(path);
        div.appendChild(removeButton);
        selectedListContainer.appendChild(div);
    });
}

async function exportSelectedDiscoveries() {
    const response = await fetch('http://127.0.0.1:8765/export', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(selectedSprites)
    });

    if (response.ok) {
        console.log('Export successful');
    } else {
        console.error('Export failed');
    }
}

document.getElementById('export-button').addEventListener('click', exportSelectedDiscoveries);

app.ticker.add((delta) => {
    app.renderer.render(app.stage);
});

loadAssets();

let ws;
function connect() {
    ws = new WebSocket("ws://127.0.0.1:8765/");
    ws.onclose = function () {
        setTimeout(function () {
            connect();
        }, 1000);
    };
    ws.onmessage = function (event) {
        ws.close();
        location.reload();
    };
}
connect();
