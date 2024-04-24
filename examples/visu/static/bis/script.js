// Code goes here
var stage = new  PIXI.Stage();
stage.scale.x = 10;
stage.scale.y = 10;
var canvas = document.getElementById('canvas');
var renderer = PIXI.autoDetectRenderer(610, 610, {view:canvas});
var g = new PIXI.Graphics();

g.lineStyle(0.2, 0xFFFF00);

g.drawPolygon([15,15,0.75*60,15,0.75*60,0.75*60,15,0.75*60]);


// const videoUrl = 'discoveries/2024-04-23T15:59_exp_0_idx_1_seed_42/870a0346300002a3a058603b28888cb4917d1cbe.discovery';
// const texture = PIXI.Texture.fromVideoUrl(videoUrl);
// const sprite = new PIXI.Sprite(texture);

// sprite.width = 300;
// sprite.height = 300;

// // Add the sprite to the stage
// stage.addChild(sprite);

// Start the PixiJS loop
animate();


stage.addChild(g);
requestAnimationFrame(animate);


function animate() {
  requestAnimationFrame(
    animate);
  renderer.render(stage);
}

function zoom(s,x,y){
 
  s = s > 0 ? 2 : 0.5;
  document.getElementById("oldScale").innerHTML = stage.scale.x.toFixed(4);
  document.getElementById("oldXY").innerHTML = '('+stage.x.toFixed(4)+','+stage.y.toFixed(4)+')';
  var worldPos = {x: (x - stage.x) / stage.scale.x, y: (y - stage.y)/stage.scale.y};
  var newScale = {x: stage.scale.x * s, y: stage.scale.y * s};
  
  var newScreenPos = {x: (worldPos.x ) * newScale.x + stage.x, y: (worldPos.y) * newScale.y + stage.y};

  stage.x -= (newScreenPos.x-x) ;
  stage.y -= (newScreenPos.y-y) ;
  stage.scale.x = newScale.x;
  stage.scale.y = newScale.y;
  document.getElementById("scale").innerHTML = newScale.x.toFixed(4);
  document.getElementById("xy").innerHTML = '('+stage.x.toFixed(4)+','+stage.y.toFixed(4)+')';
  
  document.getElementById("c").innerHTML=c;
};

var lastPos = null
$(canvas)
  .mousewheel(function(e){
  zoom(e.deltaY, e.offsetX, e.offsetY)
}).mousedown(function(e) {
  lastPos = {x:e.offsetX,y:e.offsetY};
}).mouseup(function(event) {
  lastPos = null;
}).mousemove(function(e){
  if(lastPos) {
    
    stage.x += (e.offsetX-lastPos.x);
    stage.y += (e.offsetY-lastPos.y);  
    lastPos = {x:e.offsetX,y:e.offsetY};
  }
  
});