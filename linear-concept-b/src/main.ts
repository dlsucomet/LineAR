// ─── src/main.ts ─────────────────────────────────────────────────────────────
import { CameraTracker } from './core/cameratracker';
import { HandTracker } from './core/handtracker';
import { GridRenderer } from './core/gridrender';
import { ColorOverlay } from './core/coloroverlay';
import { createCalibrationSession } from './core/calibration';

async function bootstrap() {
  console.log("Initializing LineAR Concept B...");

  // 1. Initialize the Grid (Projector View)
  const grid = new GridRenderer('jxgbox', {
    gridSize: 20,
    showLabels: true,
    animateTransition: true,
    transitionMs: 500
  });

  // 2. Initialize the p5.js Overlay (Visual Feedback)
  const overlay = new ColorOverlay();
  overlay.mount('p5-container', window.innerWidth, window.innerHeight);

  // 3. Initialize Trackers
  const videoElement = document.getElementById('video-input') as HTMLVideoElement;
  const processCanvas = document.createElement('canvas'); // Hidden canvas for OpenCV

  const cameraTracker = new CameraTracker(videoElement, processCanvas);
  const handTracker = new HandTracker();

  // 4. Start the Hand Tracker
  await handTracker.init();
  handTracker.start(videoElement);

  // 5. Start the Camera (OpenCV) Tracker
  // Note: Using 1280x720 to match your Redragon webcam's capabilities
  await cameraTracker.start({ width: 1280, height: 720, fps: 30 });

  // 6. The "Magic" Loop: Linking Input to Output
  // This runs every time the trackers detect something

  // Hand Tracking Loop
  handTracker.onFrame((hands) => {
    // Pass detected hands to p5 for drawing skeletons
    overlay.update([], hands);

    // Example: If a hand is "pinching", you could trigger a grid transformation
    if (hands.length > 0 && hands[0].gesture === 'pinch') {
      console.log("Pinch detected! Triggering transformation...");
      // grid.applyTransformation(...)
    }
  });

  // Object Tracking Loop (OpenCV)
  cameraTracker.onFrame((objects) => {
    // Pass color-coded objects to p5 overlay
    overlay.update(objects, []);

    // If you have a calibration finished, map these to the Grid:
    objects.forEach(obj => {
      // Here you would use your calibration.ts logic to map 
      // obj.center (camera px) -> grid coordinates
    });
  });
}

// Ensure the DOM is loaded and OpenCV (cv) is ready before starting
window.addEventListener('load', () => {
  const checkCV = setInterval(() => {
    if (window.cv && window.cv.Mat) {
      clearInterval(checkCV);
      bootstrap();
    }
  }, 100);
});
