import * as PIXI from 'https://unpkg.com/pixi.js@8.1.6/dist/pixi.min.mjs';

let coordinates = [];
let zoomLevel = 1;
let sprites = [];
let selected_discoveries = [];
let draggedCoordinates = { x: 0, y: 0 };
let mousePosition = { x: 0, y: 0 };

const app = new PIXI.Application();

let target_texture = await PIXI.Assets.load('/static/target.png');
let target_sprite = new PIXI.Sprite(target_texture);

//on click hide it
target_sprite.interactive = true;
target_sprite.on('click', function() {
    target_sprite.visible = false;
    // fetch get disable_target
    fetch('/disable_target', {
        method: 'GET',
    });

});

target_sprite.width = 64;
target_sprite.height = 64;
target_sprite.anchor.set(0.5);
target_sprite.visible = false;

let targetCoordinates = { x: 0, y: 0 };

//load target.json
let target_json = await PIXI.Assets.load('/discoveries/target.json');
if (target_json.detail !== "File not found") {
    targetCoordinates = target_json;
    target_sprite.visible = true;
}   



target_sprite.x = 100;
target_sprite.y = 100;

const ZOOM_FACTOR = 100;

function sceneCoordinates(x, y) {
    return {
        x: app.renderer.screen.width / 2 + zoomLevel * (ZOOM_FACTOR * x + draggedCoordinates.x) + draggedCoordinates.x,
        y: app.renderer.screen.height / 2 + zoomLevel * (ZOOM_FACTOR * y + draggedCoordinates.y) + draggedCoordinates.y
    };
}

function screenToSceneCoordinates(xScene, yScene) {
    const sceneX = (xScene - app.renderer.screen.width / 2 - draggedCoordinates.x) / zoomLevel;
    const x = (sceneX - draggedCoordinates.x) / ZOOM_FACTOR;

    const sceneY = (yScene - app.renderer.screen.height / 2 - draggedCoordinates.y) / zoomLevel;
    const y = (sceneY - draggedCoordinates.y) / ZOOM_FACTOR;

    return { x: x, y: y };
}


function computeCoordinates() {
    for (let i = 0; i < sprites.length; i++) {
        const newCoordinates = sceneCoordinates(coordinates[i].x, coordinates[i].y);
        sprites[i].position.set(newCoordinates.x, newCoordinates.y);
    }

    const newCoordinates = sceneCoordinates(targetCoordinates.x, targetCoordinates.y);
    target_sprite.position.set(newCoordinates.x, newCoordinates.y);
}

window.addEventListener('wheel', function(event) {
    zoomLevel += (-event.deltaY / ZOOM_FACTOR) * zoomLevel ** 0.5;
    zoomLevel = Math.max(1, zoomLevel);
    computeCoordinates();
});

let isDragging = false;
let lastMousePosition = { x: 0, y: 0 };

window.addEventListener('mousedown', function(event) {
    isDragging = true;
    lastMousePosition = { x: event.clientX, y: event.clientY };
});

window.addEventListener('mouseup', function(event) {
    isDragging = false;
});

window.addEventListener('mousemove', function(event) {
    if (isDragging) {
        draggedCoordinates.x += (event.clientX - lastMousePosition.x) / zoomLevel;
        draggedCoordinates.y += (event.clientY - lastMousePosition.y) / zoomLevel;
        lastMousePosition = { x: event.clientX, y: event.clientY };
        computeCoordinates();
    }
    // Update the mouse position
    const rect = app.view.getBoundingClientRect();
    mousePosition = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
});

// Add an event listener for the space key
window.addEventListener('keydown', function(event) {
    if (event.code === 'Space') {
        // Convert the current mouse position to scene coordinates
        const scenePos = screenToSceneCoordinates(mousePosition.x, mousePosition.y);
        // Move the target sprite to the scene position
        targetCoordinates = scenePos;
        target_sprite.visible = true;
        
        target_sprite.position.set(mousePosition.x, mousePosition.y);

        // post update_target with x and y
        fetch('/update_target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                x: targetCoordinates.x,
                y: targetCoordinates.y,
            }),
        });

    }
});

app.init({
    backgroundColor: 0xefffff,
    resizeTo: window
}).then(async () => {
    document.getElementById('canvas-container').appendChild(app.canvas);

    let overlayText = document.getElementById('overlay-text');
    coordinates = await PIXI.Assets.load('/static/discoveries.json');
    let concatenatedVisuals = await PIXI.Assets.load('/static/concatenated.webm');

    const numVideos = coordinates.length;
    const rows = Math.ceil(Math.sqrt(numVideos));
    const cols = Math.ceil(numVideos / rows);

    let shift_x = 0;
    let shift_y = 0;

    for (let i = 0; i < coordinates.length; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        shift_x = col * coordinates[i].width;
        shift_y = row * coordinates[i].height;

        let subTexture = new PIXI.Texture({
            source: concatenatedVisuals.source,
            frame: new PIXI.Rectangle(shift_x, shift_y, coordinates[i].width, coordinates[i].height),
        });

        subTexture.baseTexture.resource.loop = true;
        let sprite = PIXI.Sprite.from(subTexture);

        sprite.width = 128;
        sprite.height = 128;
        sprite.interactive = true;

        sprite.on('mouseover', (function(i) {
            return function() {
                overlayText.innerHTML = coordinates[i].visual;
            };
        })(i));

        sprite.on('click', (function(i) {
            return function() {
                if (selected_discoveries.includes(coordinates[i])) {
                    selected_discoveries = selected_discoveries.filter(item => item !== coordinates[i]);
                } else {
                    selected_discoveries.push(coordinates[i]);
                }
                updateSelectedSprites();
            };
        })(i));

        sprite.anchor.set(0.5);
        sprite.x = shift_x + coordinates[i].width / 2;
        sprite.y = shift_y + coordinates[i].height / 2;

        sprites.push(sprite);

        app.stage.addChild(sprite);
    }

    computeCoordinates();
    animate();

    app.stage.addChild(target_sprite);

    function animate() {
        requestAnimationFrame(animate);
        app.renderer.render(app.stage);
    }

    function updateSelectedSprites() {
        let selectedSprites = document.getElementById('selected-sprites');
        selectedSprites.innerHTML = '';
        for (let j = 0; j < selected_discoveries.length; j++) {
            const selectedSprite = document.createElement('div');
            selectedSprite.innerHTML = selected_discoveries[j].visual;
            selectedSprites.appendChild(selectedSprite);
        }
    }
});

let ws;

function connect() {
    ws = new WebSocket("ws://127.0.0.1:8765/ws");
    ws.onclose = function() {
        setTimeout(connect, 1000);
    };
    ws.onmessage = function() {
        ws.close();
        location.reload();
    };
}

connect();

document.getElementById('export-button').addEventListener('click', async () => {
    try {
        const visuals = selected_discoveries.map((coordinate) => coordinate.visual);

        let response = await fetch('/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(visuals),
        }).then((response) => response.json());

        if (response.status === 'ok') {
            // alert with new_dir
            alert('Exported to: ' + response.new_dir);
        } else {
            alert('Failed to export. Please try again.');
        }
    } catch (error) {
        console.error('Error during export:', error);
        alert('An error occurred. Please try again.');
    }
});
