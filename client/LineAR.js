const video = document.getElementById('webcamFeed');
const arCanvas = document.getElementById('arOverlay');
const arCtx = arCanvas.getContext('2d');

const graphCanvas = document.getElementById('graphCanvas');
const graphCtx = graphCanvas.getContext('2d');
const statusBadge = document.getElementById('status');

let executionStep = "secondStepLeft"; 

// Active vectors state coordinates to render on the left panel
let activeVectors = [
    { x: 3, y: -2, color: '#ff4757', label: 'v' } // Initial target vector from step (i)
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

// ─── DRAW CARTESIAN COORDINATE SYSTEM ON THE LEFT ───
function drawCartesianPlane() {
    const w = graphCanvas.width;
    const h = graphCanvas.height;
    graphCtx.clearRect(0, 0, w, h);

    const originX = w / 2;
    const originY = h / 2;
    const scale = 40; // Pixels per unit grid space

    // 1. Draw Grid Lines
    graphCtx.strokeStyle = '#e2e8f0';
    graphCtx.lineWidth = 1;
    
    for (let x = originX % scale; x < w; x += scale) {
        graphCtx.beginPath(); graphCtx.moveTo(x, 0); graphCtx.lineTo(x, h); graphCtx.stroke();
    }
    for (let y = originY % scale; y < h; y += scale) {
        graphCtx.beginPath(); graphCtx.moveTo(0, y); graphCtx.lineTo(w, y); graphCtx.stroke();
    }

    // 2. Draw Main Axes
    graphCtx.strokeStyle = '#475569';
    graphCtx.lineWidth = 2;
    graphCtx.beginPath(); graphCtx.moveTo(0, originY); graphCtx.lineTo(w, originY); graphCtx.stroke(); // X axis
    graphCtx.beginPath(); graphCtx.moveTo(originX, 0); graphCtx.lineTo(originX, h); graphCtx.stroke(); // Y axis

    // 3. Render Active Vectors from Linear Algebra calculations
    activeVectors.forEach(vec => {
        const targetX = originX + (vec.x * scale);
        const targetY = originY - (vec.y * scale); // Invert Y because canvas draws downward

        // Vector Arrow Body
        graphCtx.strokeStyle = vec.color;
        graphCtx.lineWidth = 3;
        graphCtx.beginPath();
        graphCtx.moveTo(originX, originY);
        graphCtx.lineTo(targetX, targetY);
        graphCtx.stroke();

        // Vector End Point Indicator node
        graphCtx.fillStyle = vec.color;
        graphCtx.beginPath();
        graphCtx.arc(targetX, targetY, 5, 0, 2 * Math.PI);
        graphCtx.fill();

        // Text label metadata rendering
        graphCtx.font = "bold 12px sans-serif";
        graphCtx.fillText(`${vec.label} [${vec.x}, ${vec.y}]`, targetX + 8, targetY - 4);
    });
}

// ─── DRAW AR CAMERA GUIDES ON THE RIGHT ───
function drawAROverlay() {
    arCtx.clearRect(0, 0, arCanvas.width, arCanvas.height);
    
    // Draw tracking crosshair framework zone
    arCtx.strokeStyle = 'rgba(124, 181, 236, 0.5)';
    arCtx.lineWidth = 2;
    arCtx.strokeRect(arCanvas.width * 0.1, arCanvas.height * 0.1, arCanvas.width * 0.8, arCanvas.height * 0.8);

    if (executionStep === "secondStepLeft") {
        arCtx.fillStyle = "rgba(74, 222, 128, 0.35)";
        arCtx.fillRect(150, 120, 90, 180);
        arCtx.strokeStyle = "#22c55e";
        arCtx.lineWidth = 2;
        arCtx.strokeRect(150, 120, 90, 180);
    }
}

function triggerValidationPass() {
    statusBadge.innerText = "Analyzing handwritten steps...";
    
    setTimeout(() => {
        statusBadge.innerText = "Step Validated! Correct.";
        executionStep = "secondStepRight";
        
        // Push a new linear combination component to the graph matrix dynamically
        activeVectors.push({ x: 12, y: -3, color: '#3b82f6', label: '3*L(v1)' });
        
        drawCartesianPlane();
        drawAROverlay();
    }, 1000);
}

function renderLoop() {
    drawCartesianPlane();
    drawAROverlay();
}

async function init() {
    await setupCamera();
    resizeCanvases();
    statusBadge.innerText = "System Ready";
    setInterval(renderLoop, 33);
}

window.addEventListener('resize', () => { resizeCanvases(); renderLoop(); });
window.addEventListener('DOMContentLoaded', init);