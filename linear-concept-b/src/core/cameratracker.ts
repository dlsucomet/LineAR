// ─── src/core/cameraTracker.ts ────────────────────────────────────────────────
// Manages webcam capture and uses OpenCV.js for colour-based object detection.
// OpenCV.js is loaded as a global script (window.cv) from a CDN in index.html.

import type { CameraConfig, DetectedObject, BoundingBox, Point2D } from "../types/index.js";
import { generateId } from "../utils/helpers.js";

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

/** HSV colour ranges for simple colour label detection. */
const COLOR_RANGES: Array<{
  label: string;
  hLow: number;
  hHigh: number;
  sLow: number;
  vLow: number;
}> = [
    { label: "red", hLow: 0, hHigh: 10, sLow: 100, vLow: 80 },
    { label: "red2", hLow: 160, hHigh: 180, sLow: 100, vLow: 80 },
    { label: "green", hLow: 40, hHigh: 80, sLow: 60, vLow: 60 },
    { label: "blue", hLow: 100, hHigh: 130, sLow: 80, vLow: 60 },
    { label: "yellow", hLow: 20, hHigh: 35, sLow: 100, vLow: 100 },
  ];

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
   * Detect coloured objects in the current canvas frame using HSV thresholding
   * and contour detection (OpenCV.js).
   */
  private detectObjects(): DetectedObject[] {
    const cv = window.cv;
    if (!cv) return [];

    const imageData = this.ctx.getImageData(
      0, 0, this.canvas.width, this.canvas.height,
    );

    // Convert to OpenCV Mat (RGBA → BGR)
    const src = cv.matFromImageData(imageData);
    const hsv = new cv.Mat();
    cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB, 0);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV, 0);

    const results: DetectedObject[] = [];

    for (const range of COLOR_RANGES) {
      const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(),
        [range.hLow, range.sLow, range.vLow, 0]);
      const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(),
        [range.hHigh, 255, 255, 255]);

      const mask = new cv.Mat();
      const closed = new cv.Mat();
      const kernel = cv.Mat.ones(5, 5, cv.CV_8U);

      cv.inRange(hsv, low, high, mask);
      cv.morphologyEx(mask, closed, cv.MORPH_CLOSE, kernel);

      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        if (area < MIN_CONTOUR_AREA) { contour.delete(); continue; }

        const rect = cv.boundingRect(contour);
        const bb: BoundingBox = { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
        const center: Point2D = {
          x: rect.x + rect.width / 2,
          y: rect.y + rect.height / 2,
        };

        // Map hLow → rough hue midpoint in 0-360 range
        const hue = ((range.hLow + range.hHigh) / 2) * 2; // OpenCV hue 0-180 → 0-360

        results.push({
          id: generateId(),
          boundingBox: bb,
          center,
          colorLabel: range.label === "red2" ? "red" : range.label,
          hue,
        });

        contour.delete();
      }

      // Clean up per-colour mats
      low.delete(); high.delete(); mask.delete();
      closed.delete(); kernel.delete();
      contours.delete(); hierarchy.delete();
    }

    src.delete(); hsv.delete();
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
