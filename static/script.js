const upload = document.getElementById("upload");
const preview = document.getElementById("imagePreview");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let originalImage;
let enhancedImage;

upload.addEventListener("change", function(){

const file = this.files[0];
const reader = new FileReader();

reader.onload = function(e){

preview.src = e.target.result;
originalImage = e.target.result;

}

reader.readAsDataURL(file);

});

document.getElementById("enhanceBtn").onclick = function(){

const img = new Image();
img.src = preview.src;

img.onload = function(){

canvas.width = img.width;
canvas.height = img.height;

ctx.filter = "contrast(130%) brightness(110%)";
ctx.drawImage(img,0,0);

enhancedImage = canvas.toDataURL();

preview.src = enhancedImage;

}

}

document.getElementById("compareBtn").onclick = function(){

if(preview.src === enhancedImage){

preview.src = originalImage;

}else{

preview.src = enhancedImage;

}

}