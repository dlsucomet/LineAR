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
const MIN_CONTOUR_AREA = 800;

/** Minimum area ratio for a contour to be considered a quadrilateral (0-1). */
const MIN_QUAD_RATIO = 0.35;

/** Number of vertices to look for in contour approximation (4 = quadrilateral). */
const QUAD_VERTICES = 4;

/** Epsilon factor for contour approximation (smaller = more precise). */
const APPROX_EPSILON_FACTOR = 0.03;

// ── Layer 1: Aspect Ratio ─────────────────────────────────────────────────────
/** Minimum aspect ratio (width/height) of detected quad. */
const MIN_ASPECT_RATIO = 0.35;
/** Maximum aspect ratio (width/height) of detected quad. */
const MAX_ASPECT_RATIO = 2.8;

// ── Layer 2: Frame-fraction Size ──────────────────────────────────────────────
/** Minimum fraction of frame width/height (0-1). */
const MIN_SIZE_FRACTION = 0.05;
/** Maximum fraction of frame width/height (0-1). */
const MAX_SIZE_FRACTION = 0.80;

// ── Layer 3: Multi-frame Position Consistency ────────────────────────────────
/** How many consecutive frames must have similar position before confirming. */
const POSITION_CONSISTENCY_FRAMES = 2;
/** Max pixel distance between positions to be considered "consistent". */
const POSITION_TOLERANCE_PX = 60;

// ── Layer 4: Initial Frame Lockout ────────────────────────────────────────────
/** Frames to skip at start before detection activates. */
const LOCKOUT_FRAMES = 15;

// ── Performance Throttle ───────────────────────────────────────────────────────
/** Run detection every N frames to reduce CPU load. 1 = every frame, 3 = every 3rd frame. */
const DETECTION_INTERVAL = 1;

// ---------------------------------------------------------------------------
// CameraTracker class
// ---------------------------------------------------------------------------

export class CameraTracker {
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private stream: MediaStream | null = null;
  private isRunning = false;
  private isPaused = false;
  private frameCallback: ((objects: DetectedObject[]) => void) | null = null;
  private animFrameId: number | null = null;
  private frameCount = 0;
  private recentPositions: Point2D[] = [];
  private lastConfirmedPosition: Point2D | null = null;
  private cachedObjects: DetectedObject[] = [];

  // ── Background Subtraction (projector-friendly) ────────────────────────────
  /** Accumulated background grayscale frame (null until capture completes). */
  private backgroundGray: any = null;
  /** Number of frames fed into background accumulation. */
  private bgFrameCount = 0;
  /** Number of frames to average for background. */
  private readonly BG_CAPTURE_FRAMES = 10;
  /** True once background has been fully captured. */
  private bgCaptured = false;

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
    console.log("[CameraTracker] Stream created, waiting for metadata...");
    await new Promise<void>((resolve) => {
      this.video.onloadedmetadata = () => {
        console.log("[CameraTracker] Video metadata: width=" + this.video.videoWidth + " height=" + this.video.videoHeight);
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        console.log("[CameraTracker] Canvas set to: " + this.canvas.width + "x" + this.canvas.height);
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
    this.disposeBackground();
  }

  /** Pause detection - returns empty results without running OpenCV. */
  pause(): void {
    this.isPaused = true;
    this.recentPositions = [];
    this.lastConfirmedPosition = null;
    this.disposeBackground();
    console.log("[CameraTracker] Paused");
  }

  /** Resume detection after pause. */
  resume(): void {
    this.isPaused = false;
    console.log("[CameraTracker] Resumed");
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
    this.frameCount++;

    if (this.isPaused) {
      this.frameCallback?.([]);
      this.animFrameId = requestAnimationFrame(() => this.loop());
      return;
    }

    if (this.frameCount % DETECTION_INTERVAL === 0) {
      const objects = this.detectObjects();
      this.cachedObjects = objects;
      this.frameCallback?.(objects);
    } else {
      this.frameCallback?.(this.cachedObjects);
    }

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
    if (!cv) {
      return [];
    }

    if (this.canvas.width === 0 || this.canvas.height === 0) {
      console.log("[CameraTracker] Canvas has no dimensions - skipping frame");
      return [];
    }

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

    // ── Layer 4: Background capture (replaces initial lockout) ────────────────
    // First BG_CAPTURE_FRAMES frames are used to build a clean background model.
    // This enables background subtraction so the projected grid is invisible to
    // contour detection. After capture, the foreground mask is applied before Canny.
    if (!this.bgCaptured) {
      if (this.backgroundGray === null) {
        this.backgroundGray = gray.clone();
        this.bgFrameCount = 1;
      } else {
        cv.addWeighted(this.backgroundGray, 0.9, gray, 0.1, 0, this.backgroundGray);
        this.bgFrameCount++;
      }
      if (this.bgFrameCount >= this.BG_CAPTURE_FRAMES) {
        this.bgCaptured = true;
        console.log("[CameraTracker] Background captured (" + this.BG_CAPTURE_FRAMES + " frames)");
      }
      // No detection during capture
      src.delete(); gray.delete(); edges.delete(); contours.delete(); hierarchy.delete();
      return [];
    }

    // Apply background subtraction: zero out pixels that match the background
    const diff = new cv.Mat();
    const fgMask = new cv.Mat();
    cv.absdiff(gray, this.backgroundGray, diff);
    cv.threshold(diff, fgMask, 30, 255, cv.THRESH_BINARY);
    diff.delete();
    cv.bitwise_and(gray, gray, gray, fgMask);
    fgMask.delete();

    // Apply Gaussian blur to reduce noise
    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(7, 7), 0, 0, cv.BORDER_DEFAULT);

    // Apply Canny edge detection
    cv.Canny(blurred, edges, 40, 100, 3, false);

    // Dilate edges to connect nearby contours
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    const dilated = new cv.Mat();
    cv.dilate(edges, dilated, kernel, new cv.Point(-1, -1), 3);

    // Find contours
    cv.findContours(dilated, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    // Find the largest quadrilateral (4-vertex polygon)
    let bestQuad: { contour: any; area: number; corners: Point2D[]; boundingBox: BoundingBox } | null = null;

    const debugSkip = { area: 0, ratio: 0, vertices: 0, aspect: 0, size: 0 };

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);

      // Skip small contours
      if (area < MIN_CONTOUR_AREA) { debugSkip.area++; contour.delete(); continue; }

      // Get bounding rectangle for area ratio check
      const rect = cv.boundingRect(contour);
      const rectArea = rect.width * rect.height;
      const areaRatio = area / rectArea;

      // Skip if area ratio suggests it's not a solid shape
      if (areaRatio < MIN_QUAD_RATIO) { debugSkip.ratio++; contour.delete(); continue; }

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

        // Calculate bounding box from corners
        const xs = corners.map(c => c.x);
        const ys = corners.map(c => c.y);
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

        // ── Layer 1: Aspect ratio filter ──────────────────────────────────────────
        const aspectRatio = bb.width / bb.height;
        if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) {
          debugSkip.aspect++;
          approx.delete();
          contour.delete();
          continue;
        }

        // ── Layer 2: Frame-fraction size filter ───────────────────────────────────
        const fw = this.canvas.width;
        const fh = this.canvas.height;
        const widthFraction = bb.width / fw;
        const heightFraction = bb.height / fh;
        if (widthFraction < MIN_SIZE_FRACTION || widthFraction > MAX_SIZE_FRACTION ||
            heightFraction < MIN_SIZE_FRACTION || heightFraction > MAX_SIZE_FRACTION) {
          debugSkip.size++;
          approx.delete();
          contour.delete();
          continue;
        }

        // Keep the largest quadrilateral
        if (!bestQuad || area > bestQuad.area) {
          if (bestQuad) bestQuad.contour.delete();
          bestQuad = { contour, area, corners, boundingBox: bb };
        } else {
          contour.delete();
        }
      } else {
        debugSkip.vertices++;
        contour.delete();
      }

      approx.delete();
    }

    // ── Layer 3: Multi-frame position consistency ─────────────────────────────
    const results: DetectedObject[] = [];
    if (bestQuad) {
      const center: Point2D = {
        x: bestQuad.boundingBox.x + bestQuad.boundingBox.width / 2,
        y: bestQuad.boundingBox.y + bestQuad.boundingBox.height / 2,
      };

      // Add to recent positions
      this.recentPositions.push(center);
      if (this.recentPositions.length > POSITION_CONSISTENCY_FRAMES) {
        this.recentPositions.shift();
      }

      // Check if positions are consistent
      let isConsistent = false;
      if (this.lastConfirmedPosition) {
        // Check if recent positions are close to last confirmed
        const avgDist = this.recentPositions.reduce((sum, p) => {
          const dx = p.x - this.lastConfirmedPosition!.x;
          const dy = p.y - this.lastConfirmedPosition!.y;
          return sum + Math.sqrt(dx * dx + dy * dy);
        }, 0) / this.recentPositions.length;
        isConsistent = avgDist <= POSITION_TOLERANCE_PX;
      } else if (this.recentPositions.length >= POSITION_CONSISTENCY_FRAMES) {
        // Check if recent positions are close to each other
        const ref = this.recentPositions[0]!;
        isConsistent = this.recentPositions.every(p => {
          const dx = p.x - ref.x;
          const dy = p.y - ref.y;
          return Math.sqrt(dx * dx + dy * dy) <= POSITION_TOLERANCE_PX;
        });
      }

      if (isConsistent || this.recentPositions.length < POSITION_CONSISTENCY_FRAMES) {
        // Update confirmed position
        this.lastConfirmedPosition = {
          x: this.recentPositions.reduce((s, p) => s + p.x, 0) / this.recentPositions.length,
          y: this.recentPositions.reduce((s, p) => s + p.y, 0) / this.recentPositions.length,
        };

        console.log("[CameraTracker] Object detected (pending confirm):", this.recentPositions.length + "/" + POSITION_CONSISTENCY_FRAMES, "frames");
        if (this.recentPositions.length >= POSITION_CONSISTENCY_FRAMES) {
          console.log("[CameraTracker] Object CONFIRMED:", bestQuad.boundingBox);
        }
      } else {
        // Positions not consistent - reset
        this.recentPositions = [];
        console.log("[CameraTracker] Position inconsistent, resetting tracking");
      }

      bestQuad.contour.delete();
    } else {
      // No quad found - reset tracking if no detections for a while
      if (this.frameCount % 30 === 0) {
        this.recentPositions = [];
        this.lastConfirmedPosition = null;
      }
    }

    // Clean up all OpenCV Mats
    src.delete(); gray.delete(); edges.delete(); blurred.delete();
    dilated.delete(); kernel.delete(); contours.delete(); hierarchy.delete();

    // Only return confirmed objects
    if (bestQuad && this.recentPositions.length >= POSITION_CONSISTENCY_FRAMES) {
      results.push({
        id: generateId(),
        boundingBox: bestQuad.boundingBox,
        center: {
          x: bestQuad.boundingBox.x + bestQuad.boundingBox.width / 2,
          y: bestQuad.boundingBox.y + bestQuad.boundingBox.height / 2,
        },
        colorLabel: "detected",
        hue: 0,
      });
    }

    return results;
  }

  // -------------------------------------------------------------------------
  // Utility
  // -------------------------------------------------------------------------

  /** Clean up background subtraction resources. */
  private disposeBackground(): void {
    if (this.backgroundGray) {
      try { this.backgroundGray.delete(); } catch { /* ignore */ }
      this.backgroundGray = null;
    }
    this.bgFrameCount = 0;
    this.bgCaptured = false;
  }

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
