const video = document.getElementById('webcamFeed');
const canvas = document.getElementById('arOverlay');
const ctx = canvas.getContext('2d');
const statusBadge = document.getElementById('status');

// Target answer parameters for Problem 1
let executionStep = "secondStepLeft"; 
const validationTargets = {
    "secondStepLeft": ["12", "-3"],
    "secondStepRight": ["2", "-4"],
    "finalAnswer": ["14", "-7"]
};

// Initialize Web Video Components
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1920, height: 1080, facingMode: "environment" },
            audio: false
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } catch (err) {
        statusBadge.innerText = "Error: Camera Access Denied";
        console.error(err);
    }
}

function configureCanvasSize() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
}

// Draws static user targets matching your design layout
function renderTargetGuidelines() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw Target Alignment Guides (Simulating bounding tracking ranges)
    ctx.strokeStyle = '#7cb5ec';
    ctx.lineWidth = 4;
    ctx.strokeRect(canvas.width * 0.25, canvas.height * 0.15, canvas.width * 0.5, canvas.height * 0.7);

    // Conditional AR Verification Projection Boxes
    if (executionStep === "secondStepLeft") {
        ctx.fillStyle = "rgba(74, 222, 128, 0.4)"; // Projected translucent target highlight
        ctx.fillRect(320, 220, 80, 140);
        
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 3;
        ctx.strokeRect(320, 220, 80, 140);
    }
}

// Emulates your pipeline execution flow inside the browser
function triggerValidationPass() {
    statusBadge.innerText = "Processing Step OCR...";
    
    // Emulating an OCR evaluation hit cycle against current target coordinates
    setTimeout(() => {
        // Mocking positive handwriting detection evaluation from the OCR cycle
        statusBadge.innerText = "Step Validated: Correct!";
        
        // Transition system processing states to update target tracking locations
        executionStep = "secondStepRight";
        renderTargetGuidelines();
    }, 1200);
}

// Core App Initialization Routine
async function init() {
    await setupCamera();
    configureCanvasSize();
    statusBadge.innerText = "System Status: Active";
    
    // Run regular frame overlay update loops
    setInterval(renderTargetGuidelines, 33); // Handles redraw frames approx 30fps
}

// Global Event Triggers
window.addEventListener('resize', configureCanvasSize);
window.addEventListener('DOMContentLoaded', init);