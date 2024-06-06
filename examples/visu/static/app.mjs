import * as PIXI from 'https://unpkg.com/pixi.js@8.x/dist/pixi.min.mjs';



let coordinates=[];
 let zoomLevel = 1;

let  sprites = [];

//set
let selected_discoveries = [];



let draggedCoordinates = {x:0,y:0};

const ZOOM_FACTOR=100;

function computeCoordinates(){
    for (let i = 0; i < sprites.length; i++) {
        sprites[i].position.set(
          (
             app.renderer.screen.width/2 +
             
             zoomLevel  * (ZOOM_FACTOR*coordinates[i].x
                + draggedCoordinates.x) + draggedCoordinates.x
             )

                ,
             (app.renderer.screen.height /2+zoomLevel* (ZOOM_FACTOR*coordinates[i].y
                + draggedCoordinates.y) + draggedCoordinates.y)
              
        );


        console.log(sprites[i].position.x);
        
    }
}

window.addEventListener('wheel', function(event) {
    zoomLevel += (-event.deltaY /ZOOM_FACTOR)*zoomLevel**0.5;

    //zoom on pointer position


    zoomLevel = Math.max(1,zoomLevel);

    
    computeCoordinates();

});

//dragg&drop to move the draggedCoordinates
let isDragging = false;
let lastMousePosition = {x:0,y:0};
window.addEventListener('mousedown', function(event) {
    isDragging = true;
    lastMousePosition = {x:event.clientX,y:event.clientY};
});
window.addEventListener('mouseup', function(event) {
    isDragging = false;
});
window.addEventListener('mousemove', function(event) {
    if(isDragging){
        draggedCoordinates.x += (event.clientX - lastMousePosition.x)/zoomLevel
        draggedCoordinates.y += (event.clientY - lastMousePosition.y)/zoomLevel
        lastMousePosition = {x:event.clientX,y:event.clientY};
        computeCoordinates();
    }
});


//background: "#1099bb"
const app = new PIXI.Application();
app
  .init({ 
    backgroundColor: 0xefffff,
    resizeTo: window})
  .then(async () => {



    document.getElementById('canvas-container').appendChild(app.canvas);

    var overlayText=document.getElementById('overlay-text');


    //load json file
    // const json = await fetch('/discoveries.json')
    // console.log(json);

    coordinates= await PIXI.Assets.load('/discoveries.json')

    console.log(coordinates);

    let concatenatedVisuals = await PIXI.Assets.load('concatenated.webm');

    



    // var texture = await PIXI.Assets.load(
    //     {src:'/discoveries/2024-04-23T15:59_exp_0_idx_0_seed_42/68daef6e9648f282ba85406124a8412ee6645f4c.mp4',
    //     loader: 'loadTextures',
    //     });

    


// Calculate the number of rows and columns needed to form a grid
const numVideos = coordinates.length;
const rows = Math.ceil(Math.sqrt(numVideos));
const cols = Math.ceil(numVideos / rows);

let shift_x = 0;
let shift_y = 0;

for (let i = 0; i < coordinates.length; i++) {
    // Calculate the row and column for the current video
    const row = Math.floor(i / cols);
    const col = i % cols;

    // Calculate the x and y positions based on the row and column
    shift_x = col * coordinates[i].width;
    shift_y = row * coordinates[i].height;

    var subTexture = new PIXI.Texture({
        source: concatenatedVisuals.source,
        frame: new PIXI.Rectangle(shift_x, shift_y, coordinates[i].width, coordinates[i].height),
    });

  
    
    subTexture.baseTexture.resource.loop = true;

    var sprite = PIXI.Sprite.from(subTexture);

    sprite.width = 128;
    sprite.height = 128;

    

    sprite.interactive = true;
    sprite.on('mouseover', (function(i) {
        return function() {
            // Display the name of the visual
            overlayText.innerHTML = coordinates[i].visual;
        };
    })(i));

    //onclick append to selected discoveries
    sprite.on('click', (function(i) {
        return function() {
            // Display the name of the visual
            //if already in the list, remove it
            if (selected_discoveries.includes(coordinates[i])) {
                selected_discoveries = selected_discoveries.filter(item => item !== coordinates[i]);
            } else {
            selected_discoveries.push(coordinates[i]);
            }
            // refresh selected discoveries list in the overlay
            //in div id=selected-sprites
            let selectedSprites = document.getElementById('selected-sprites');
            selectedSprites.innerHTML = '';
            for (let j = 0; j < selected_discoveries.length; j++) {
                const selectedSprite = document.createElement('div');
                selectedSprite.innerHTML = selected_discoveries[j].visual;
                selectedSprites.appendChild(selectedSprite);
            }
        };
    })(i));

    sprite.anchor.set(0.5);
    sprite.x = shift_x + coordinates[i].width / 2;
    sprite.y = shift_y + coordinates[i].height / 2;
    
    sprites.push(sprite);
    app.stage.addChild(sprite);
}



    computeCoordinates();






    const animate = () => {
      requestAnimationFrame(animate);
      app.renderer.render(app.stage);
    };


    animate();

    



  });


let ws;

function connect() {
  ws = new WebSocket("ws://127.0.0.1:8765/");
  ws.onclose = function() {
    setTimeout(function() {
      connect();
    }, 1000);
  };
  ws.onmessage = function(event) {
    //clean three.js scene
//reload the page
//exit the websocket
    ws.close();
    location.reload(); 
  };
}

connect();

  

