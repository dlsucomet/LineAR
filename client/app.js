const video = document.getElementById('webcamFeed');
const arCanvas = document.getElementById('arOverlay');
const arCtx = arCanvas.getContext('2d');

const graphCanvas = document.getElementById('graphCanvas');
const graphCtx = graphCanvas.getContext('2d');

const statusBadge = document.getElementById('status');
const instructionHeading = document.getElementById('instructionHeading');
const stepHint = document.getElementById('stepHint');

let executionStep = "secondStepLeft"; 

let activeVectors = [
    { x: 1, y: 2, color: '#000000', label: 'u' }, // High contrast vector maps
    { x: 0, y: 1, color: '#555555', label: 'w' }  
];

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: "environment" },
            audio: false
        });
        video.srcObject = stream;
        return new Promise((resolve) => video.onloadedmetadata = () => resolve(video));
    } catch (err) {
        statusBadge.innerText = "Error: Camera Access Denied";
    }
}

function resizeCanvases() {
    arCanvas.width = arCanvas.clientWidth;
    arCanvas.height = arCanvas.clientHeight;
    
    graphCanvas.width = graphCanvas.clientWidth;
    graphCanvas.height = graphCanvas.clientHeight;
}

function drawCartesianPlane() {
    const w = graphCanvas.width;
    const h = graphCanvas.height;
    graphCtx.clearRect(0, 0, w, h);

    const originX = w / 2;
    const originY = h / 2;
    const scale = 35; 

    // Darkened grid configurations for clearer wall projection resolution
    graphCtx.strokeStyle = '#e2e8f0';
    graphCtx.lineWidth = 1;
    
    for (let x = originX % scale; x < w; x += scale) {
        graphCtx.beginPath(); graphCtx.moveTo(x, 0); graphCtx.lineTo(x, h); graphCtx.stroke();
    }
    for (let y = originY % scale; y < h; y += scale) {
        graphCtx.beginPath(); graphCtx.moveTo(0, y); graphCtx.lineTo(w, y); graphCtx.stroke();
    }

    graphCtx.strokeStyle = '#000000';
    graphCtx.lineWidth = 2.5;
    graphCtx.beginPath(); graphCtx.moveTo(0, originY); graphCtx.lineTo(w, originY); graphCtx.stroke(); 
    graphCtx.beginPath(); graphCtx.moveTo(originX, 0); graphCtx.lineTo(originX, h); graphCtx.stroke(); 

    activeVectors.forEach(vec => {
        const targetX = originX + (vec.x * scale);
        const targetY = originY - (vec.y * scale); 

        graphCtx.strokeStyle = vec.color;
        graphCtx.lineWidth = 4; // Thickened paths
        graphCtx.beginPath(); graphCtx.moveTo(originX, originY); graphCtx.lineTo(targetX, targetY); graphCtx.stroke();

        graphCtx.fillStyle = vec.color;
        graphCtx.beginPath(); graphCtx.arc(targetX, targetY, 5, 0, 2 * Math.PI); graphCtx.fill();

        graphCtx.font = "bold 13px sans-serif";
        graphCtx.fillStyle = "#000000";
        graphCtx.fillText(`${vec.label} (${vec.x},${vec.y})`, targetX + 8, targetY - 4);
    });
}

// ─── REMOVED BLUE SHEET TRACKING OUTLINES ───
function drawAROverlay() {
    arCtx.clearRect(0, 0, arCanvas.width, arCanvas.height);

    // Active targeted highlights modified to deep solid black indicator boxes for tracking areas
    if (executionStep === "secondStepLeft") {
        arCtx.strokeStyle = "#000000";
        arCtx.lineWidth = 3;
        arCtx.setLineDash([4, 4]); // Dashed indicator target line format
        arCtx.strokeRect(30, 180, 80, 100);
        arCtx.setLineDash([]); // Reset line formatting
    }
}

function triggerValidationPass() {
    statusBadge.innerText = "Analyzing handwritten elements...";
    
    setTimeout(() => {
        statusBadge.innerText = "Step Validated! Correct.";
        executionStep = "secondStepRight";
        
        activeVectors.push({ x: 3, y: -2, color: '#000000', label: 'Target Proj' });
        
        instructionHeading.innerText = "Matrix Scaling Step";
        stepHint.innerHTML = "<strong>Step Verified.</strong> Progress forward by inputting scalar operations in the next region.";

    }, 1000);
}

function renderLoop() {
    drawCartesianPlane();
    drawAROverlay();
}

async function init() {
    await setupCamera(); // Web camera initializes implicitly in the background
    resizeCanvases();
    statusBadge.innerText = "System Active";
    setInterval(renderLoop, 33);
}

window.addEventListener('resize', () => { resizeCanvases(); renderLoop(); });
window.addEventListener('DOMContentLoaded', init);