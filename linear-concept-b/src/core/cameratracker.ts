// ─── src/core/cameraTracker.ts ────────────────────────────────────────────────
// Manages webcam capture and uses OpenCV.js for contour-based object detection.
// Detects any rectangular shape (any color) using edge/contour detection.
// OpenCV.js is loaded as a global script (window.cv) from a CDN in index.html.

import type { CameraConfig, DetectedObject, BoundingBox, Point2D } from "../types/index.ts";
import { generateId } from "../utils/helpers.ts";

// ---------------------------------------------------------------------------
// OpenCV.js ambient type shim
// (The full OpenCV.js type definitions are not always available via npm.)
// ---------------------------------------------------------------------------
declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cv: any;
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Minimum contour area (px²) to be considered a valid object. */
const MIN_CONTOUR_AREA = 1500;

/** Minimum area ratio for a contour to be considered a quadrilateral (0-1). */
const MIN_QUAD_RATIO = 0.4;

/** Number of vertices to look for in contour approximation (4 = quadrilateral). */
const QUAD_VERTICES = 4;

/** Epsilon factor for contour approximation (smaller = more precise). */
const APPROX_EPSILON_FACTOR = 0.02;

// ---------------------------------------------------------------------------
// CameraTracker class
// ---------------------------------------------------------------------------

export class CameraTracker {
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private stream: MediaStream | null = null;
  private isRunning = false;
  private frameCallback: ((objects: DetectedObject[]) => void) | null = null;
  private animFrameId: number | null = null;

  constructor(
    videoEl: HTMLVideoElement,
    canvasEl: HTMLCanvasElement,
  ) {
    this.video = videoEl;
    this.canvas = canvasEl;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) throw new Error("Could not get 2D context from canvas.");
    this.ctx = ctx;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /** Start the camera feed with the given configuration. */
  async start(config: CameraConfig): Promise<void> {
    const constraints: MediaStreamConstraints = {
      video: {
        width: { ideal: config.width },
        height: { ideal: config.height },
        frameRate: { ideal: config.fps },
        ...(config.deviceId ? { deviceId: { exact: config.deviceId } } : {}),
      },
      audio: false,
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;
    await new Promise<void>((resolve) => {
      this.video.onloadedmetadata = () => {
        this.video.play();
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        resolve();
      };
    });

    this.isRunning = true;
    this.loop();
  }

  /** Stop the camera and clean up resources. */
  stop(): void {
    this.isRunning = false;
    if (this.animFrameId !== null) cancelAnimationFrame(this.animFrameId);
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
  }

  /**
   * Register a callback that receives detected objects every frame.
   * Pass null to unregister.
   */
  onFrame(cb: ((objects: DetectedObject[]) => void) | null): void {
    this.frameCallback = cb;
  }

  // -------------------------------------------------------------------------
  // Main loop
  // -------------------------------------------------------------------------

  private loop(): void {
    if (!this.isRunning) return;

    this.ctx.drawImage(this.video, 0, 0);
    const objects = this.detectObjects();
    this.frameCallback?.(objects);

    this.animFrameId = requestAnimationFrame(() => this.loop());
  }

  // -------------------------------------------------------------------------
  // Object detection via OpenCV.js
  // -------------------------------------------------------------------------

  /**
   * Detect objects in the current canvas frame using contour/edge detection.
   * Finds any rectangular shape (any color) by detecting contours and
   * approximating them to quadrilaterals.
   */
  private detectObjects(): DetectedObject[] {
    const cv = window.cv;
    if (!cv) return [];

    const imageData = this.ctx.getImageData(
      0, 0, this.canvas.width, this.canvas.height,
    );

    // Convert to OpenCV Mat
    const src = cv.matFromImageData(imageData);
    const gray = new cv.Mat();
    const edges = new cv.Mat();
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();

    // Convert to grayscale
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

    // Apply Gaussian blur to reduce noise
    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

    // Apply Canny edge detection
    cv.Canny(blurred, edges, 50, 150, 3, false);

    // Dilate edges to connect nearby contours
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    const dilated = new cv.Mat();
    cv.dilate(edges, dilated, kernel, new cv.Point(-1, -1), 2);

    // Find contours
    cv.findContours(dilated, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    // Find the largest quadrilateral (4-vertex polygon)
    let bestQuad: { contour: any; area: number; corners: Point2D[] } | null = null;

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);

      // Skip small contours
      if (area < MIN_CONTOUR_AREA) { contour.delete(); continue; }

      // Get bounding rectangle for area ratio check
      const rect = cv.boundingRect(contour);
      const rectArea = rect.width * rect.height;
      const areaRatio = area / rectArea;

      // Skip if area ratio suggests it's not a solid shape
      if (areaRatio < MIN_QUAD_RATIO) { contour.delete(); continue; }

      // Approximate contour to a polygon
      const perimeter = cv.arcLength(contour, true);
      const epsilon = perimeter * APPROX_EPSILON_FACTOR;
      const approx = new cv.Mat();
      cv.approxPolyDP(contour, approx, epsilon, true);

      // Check if we have 4 vertices (quadrilateral)
      if (approx.rows === QUAD_VERTICES) {
        // Extract corner points
        const corners: Point2D[] = [];
        for (let j = 0; j < approx.rows; j++) {
          corners.push({
            x: approx.data32S[j * 2],
            y: approx.data32S[j * 2 + 1],
          });
        }

        // Keep the largest quadrilateral
        if (!bestQuad || area > bestQuad.area) {
          if (bestQuad) bestQuad.contour.delete();
          bestQuad = { contour, area, corners };
        } else {
          contour.delete();
        }
      } else {
        contour.delete();
      }

      approx.delete();
    }

    // If we found a quadrilateral, return it as a detected object
    const results: DetectedObject[] = [];
    if (bestQuad) {
      // Calculate bounding box from corners
      const xs = bestQuad.corners.map(c => c.x);
      const ys = bestQuad.corners.map(c => c.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);

      const bb: BoundingBox = {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
      };

      const center: Point2D = {
        x: minX + bb.width / 2,
        y: minY + bb.height / 2,
      };

      results.push({
        id: generateId(),
        boundingBox: bb,
        center,
        colorLabel: "detected",
        hue: 0,
      });

      console.log("[CameraTracker] Object detected! Corners:", bestQuad.corners);

      bestQuad.contour.delete();
    }

    // Clean up all OpenCV Mats
    src.delete(); gray.delete(); edges.delete(); blurred.delete();
    dilated.delete(); kernel.delete(); contours.delete(); hierarchy.delete();

    return results;
  }

  // -------------------------------------------------------------------------
  // Utility
  // -------------------------------------------------------------------------

  /** Return available video input devices. */
  static async listCameras(): Promise<MediaDeviceInfo[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === "videoinput");
  }

  get videoElement(): HTMLVideoElement {
    return this.video;
  }

  get resolution(): { width: number; height: number } {
    return { width: this.canvas.width, height: this.canvas.height };
  }
}
