window.addEventListener('load', ()=>{
        
    resize(); // Resizes the canvas once the window loads
    document.addEventListener('mousedown', startPainting);
    document.addEventListener('mouseup', stopPainting);
    document.addEventListener('mousemove', sketch);
    window.addEventListener('resize', resize);
});
    
const canvas = document.querySelector('#canvas');
const answer = document.querySelector('#result');

//const model = tf.loadLayersModel('./first_cnn/tfjsmodel/model.json');
   
// Context for the canvas for 2 dimensional operations
const ctx = canvas.getContext('2d');
    
// Resizes the canvas to the available size of the window.
function resize(){
  ctx.canvas.width = window.innerHeight/3;
  ctx.canvas.height = window.innerHeight/3;
}
    
// Stores the initial position of the cursor
let coord = {x:0 , y:0}; 
   
// This is the flag that we are going to use to 
// trigger drawing
let paint = false;
    
// Updates the coordianates of the cursor when 
// an event e is triggered to the coordinates where 
// the said event is triggered.
function getPosition(event){
  coord.x = event.clientX - canvas.offsetLeft;
  coord.y = event.clientY - canvas.offsetTop;
}

//clear canvas
function clear(){
    ctx.clearRect(0,0,window.innerWidth, window.innerHeight)

}
  
// The following functions toggle the flag to start
// and stop drawing
function startPainting(event){
  paint = true;
  getPosition(event);
  clear();
}
function stopPainting(){
  paint = false;
  //predict digit
  predict();
  
}
    
function sketch(event){
  if (!paint) return;
  ctx.beginPath();
    
  ctx.lineWidth = 18;
   
  // Sets the end of the lines drawn
  // to a round shape.
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#111111';
      
  // The cursor to start drawing
  // moves to this coordinate
  ctx.moveTo(coord.x, coord.y);
   
  // The position of the cursor
  // gets updated as we move the
  // mouse around.
  getPosition(event);
   
  // A line is traced from start
  // coordinate to this coordinate
  ctx.lineTo(coord.x , coord.y);
    
  // Draws the line.
  ctx.stroke();
}

async function predict(){
    const scaled = ctx.drawImage(canvas, 0, 0, 28, 28);
    //console.log(scaled)
    let img = tf.browser.fromPixels(ctx.getImageData(0, 0, 28, 28), 1);
    img = img.reshape([1, 28, 28, 1]);
    img = tf.cast(img, 'float32');
    //const output = model.predict(img);
    tf.loadLayersModel('./first_cnn/notSeqv5_20epochs_high_acc_maybe/model.json').then(function(model){
      let output = model.predict(img);
      //console.log(output);
      let predictions = Array.from(output.dataSync()); 
      console.log(predictions);
      let res = .0;
      for (let index = 0; index < 10; index++) {
        if(predictions[index] > .8) res = index;
      }
      console.log(res);
      answer.innerText = res
    })

    // Save predictions on the component
}