// Create the scene and a camera to view it
var scene = new THREE.Scene();

scene.background = new THREE.Color( 0xffffff );



/**
* Camera
**/

var refresh=false;
var askrefresh=false;


// Specify the portion of the scene visiable at any time (in degrees)
var fieldOfView = 75;

// Specify the camera's aspect ratio
var aspectRatio = window.innerWidth / window.innerHeight;

var nearPlane = 100;
var farPlane = 50000;

// Use the values specified above to create a camera
var camera = new THREE.PerspectiveCamera(
  fieldOfView, aspectRatio, nearPlane, farPlane
);

// Finally, set the camera's position

camera.position.y = 0;
camera.position.x = 0;

/**
* Renderer
**/

// Create the canvas with a renderer
var renderer = new THREE.WebGLRenderer({ antialias: true });

// Add support for retina displays
renderer.setPixelRatio( window.devicePixelRatio );

// Specify the size of the canvas
renderer.setSize( window.innerWidth, window.innerHeight );

// Add the canvas to the DOM
document.body.appendChild( renderer.domElement );

/**
* Load External Data
**/

// Create a store for image position information
var imagePositions = null;


  var loader = new THREE.FileLoader();


  var image_loader = new THREE.TextureLoader();

// Create a store for each of the 5 image atlas files
// The keys will represent the index position of the atlas file,
// and the values will contain the material itself
var materials = {};

function init() {



loader.load('discoveries.json', function(data) {
  imagePositions = JSON.parse(data);

  camera.position.z = Math.sqrt(imagePositions.length)*1000;

  //iterate over the imagePositions and create a new geometry for each
  for (var i=0; i<imagePositions.length; i++) {
   
    //load the image
    var visual=  "discoveries/"+imagePositions[i].visual;

    var material= null;


    if (imagePositions[i].mimetype.includes('video')) {
    


     
        
      var video = document.createElement('video');
      video.src = visual;
      video.autoplay = true;
      video.loop = true;
      video.muted = true;
      video.play();

      

      var texture = new THREE.VideoTexture(video);
       material = new THREE.MeshBasicMaterial({ map: texture });

      
      
      
    } else if (imagePositions[i].mimetype.includes('image')) {
      var texture = image_loader.load(visual);
       material = new THREE.MeshBasicMaterial({ map: texture });

    }

    if (material!=null) {

      var geometry = new THREE.Geometry();
      // Retrieve the x, y, z coords for this subimage
      var coords = getCoords(i);
      // Add the vertices for the image
      geometry = new THREE.PlaneGeometry( imageWidth, imageHeight);
      // Add the faces for the image
      // Create a new material for the image
  
      
      material.map.minFilter = THREE.LinearFilter;
      // Create a new mesh for the image
      var mesh = new THREE.Mesh(geometry, material);

      mesh.position.set(coords.x, coords.y, coords.z);
      // Add the mesh to the scene
        
      
      

   
        
            scene.add(mesh);


      
      
      

    }




 }



 })




 



}

init();





/**
* Load Atlas Textures
**/




/**
* Build Image Geometry
**/

const imageWidth = 256;
const imageHeight = 256;




// Get the x, y, z coords for the subimage at index position j
// of atlas in index position i
function getCoords(i) {
  var coords = imagePositions[i];
  coords.x *= ( window.innerWidth - imageWidth)*(Math.sqrt(imagePositions.length))


  coords.y *= ( window.innerHeight- imageHeight)*(Math.sqrt(imagePositions.length))
  
  coords.z = (-200 + i/10);
  return coords;
}




/**
* Lights
**/

// Add a point light with #fff color, .7 intensity, and 0 distance
var light = new THREE.PointLight( 0xffffff, 1, 0 );

// Specify the light's position
light.position.set(1, 1, 100);

// Add the light to the scene
scene.add(light)

/**
* Add Controls
**/

var controls = new THREE.TrackballControls(camera, renderer.domElement);

//disable rotation
controls.noRotate = true;

/**
* Handle window resizes
**/

window.addEventListener('resize', function() {
  if (refresh) {
    return;
  }
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize( window.innerWidth, window.innerHeight );
  controls.handleResize();
});

/**
* Render!
**/


var raycaster = new THREE.Raycaster(); 
var mouse = new THREE.Vector3(); 

var selected=null;

function onMouseClick( event ) { 
  // calculate mouse position in normalized device coordinates 
  // (-1 to +1) for both components 

  if (refresh) {
    return;
  }

   mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1; 
   mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1; 

   raycaster.setFromCamera( mouse, camera ); 

   var intersects = raycaster.intersectObjects( scene.children );

  if (intersects.length > 0) {
    if (selected!=null) {
      selected.material.color.set(0xffffff);

    }
    if (selected==intersects[0].object) {
      selected=null;
      return;
    }

    intersects[0].object.material.color.set(
      0x7DFF7D
      
      );
    selected = intersects[0].object;
  }


  //get intersected objects


} 


window.addEventListener( 'click', onMouseClick, false );        

//infinite loop to try to reconnect if the connection is closed


function connect() {
  ws = new WebSocket("ws://127.0.0.1:8765/");
  ws.onclose = function() {
    setTimeout(function() {
      connect();
    }, 1000);
  };
  ws.onmessage = function(event) {
    ws.close();
    window.location.reload();
  };
}

connect();






// The main animation function that re-renders the scene each animation frame
function animate() {




  requestAnimationFrame( animate );


  renderer.render( scene, camera );
  controls.update();
  console.log("refresh", refresh)
  }



 


animate();