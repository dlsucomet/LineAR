// ─── src/core/handTracker.ts ──────────────────────────────────────────────────
// Wraps @tensorflow-models/hand-pose-detection to detect hands and classify
// basic gestures from a video element.

import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-wasm";
import * as tf from "@tensorflow/tfjs-core";
import type { DetectedHand, HandGesture, HandLandmark } from "../types/index.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HandTrackerConfig {
  /** Use the "lite" model for faster but less accurate tracking. */
  lite?: boolean;
  /** Maximum number of hands to track. */
  maxHands?: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default maximum number of hands to track simultaneously. */
const DEFAULT_MAX_HANDS = 2;

/** Pinch distance threshold (normalised units). */
const PINCH_THRESHOLD = 0.09;

/** Number of consecutive frames a gesture must hold before being classified. */
const GESTURE_HOLD_FRAMES = 2;

// ---------------------------------------------------------------------------
// HandTracker class
// ---------------------------------------------------------------------------

export class HandTracker {
  private detector: handPoseDetection.HandDetector | null = null;
  private isRunning = false;
  private animFrameId: number | null = null;
  private frameCallback: ((hands: DetectedHand[]) => void) | null = null;
  private frameCount = 0;
  private readonly FRAME_SKIP = 1;
  private inferenceBusy = false;
  private videoEl: HTMLVideoElement | null = null;
  private config: HandTrackerConfig;

  constructor(config: HandTrackerConfig = {}) {
    this.config = config;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /** Initialise the TensorFlow model (downloads weights). */
  async init(): Promise<void> {
    tf.env().set("WEBGL_CPU_FORWARD", false);
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig: handPoseDetection.MediaPipeHandsTfjsModelConfig = {
      runtime: "tfjs",
      modelType: this.config.lite ? "lite" : "full",
      maxHands: this.config.maxHands ?? DEFAULT_MAX_HANDS,
    };
    this.detector = await handPoseDetection.createDetector(model, detectorConfig);
    console.log("[HandTracker] TF.js backend:", tf.getBackend());
  }

  /** Begin tracking hands in the given video element. */
  start(videoEl: HTMLVideoElement): void {
    if (!this.detector) {
      throw new Error("HandTracker not initialised. Call init() first.");
    }
    this.videoEl = videoEl;
    this.isRunning = true;
    this.loop();
  }

  /** Stop tracking and free resources. */
  stop(): void {
    this.isRunning = false;
    if (this.animFrameId !== null) cancelAnimationFrame(this.animFrameId);
    this.detector?.dispose();
    this.detector = null;
  }

  /** Register a callback that fires every frame with detected hands. */
  onFrame(cb: ((hands: DetectedHand[]) => void) | null): void {
    this.frameCallback = cb;
  }

  // -------------------------------------------------------------------------
  // Main loop
  // -------------------------------------------------------------------------

  private loop(): void {
    if (!this.isRunning || !this.videoEl || !this.detector) return;

    this.frameCount++;

    // Skip if previous inference hasn't finished yet (prevents overlapping promises)
    if (this.inferenceBusy) {
      this.animFrameId = requestAnimationFrame(() => this.loop());
      return;
    }

    if (this.frameCount % this.FRAME_SKIP !== 0) {
      this.animFrameId = requestAnimationFrame(() => this.loop());
      return;
    }

    this.inferenceBusy = true;
    this.detector
      .estimateHands(this.videoEl, { flipHorizontal: false })
      .then((rawHands) => {
        this.inferenceBusy = false;
        const hands: DetectedHand[] = rawHands.map((h): DetectedHand => {
          const toLandmark = (kp: { x: number; y: number; z?: number; name?: string }): HandLandmark => ({
            x: kp.x,
            y: kp.y,
            z: kp.z ?? 0,
            ...(kp.name !== undefined ? { name: kp.name } : {}),
          });

          const landmarks = h.keypoints.map(toLandmark);

          return {
            handedness: h.handedness as "Left" | "Right",
            score: h.score ?? 0,
            landmarks,
            gesture: classifyGesture(landmarks),
          };
        });

        this.frameCallback?.(hands);
        this.animFrameId = requestAnimationFrame(() => this.loop());
      })
      .catch((err) => {
        this.inferenceBusy = false;
        console.warn("[HandTracker] Frame error:", err);
        this.animFrameId = requestAnimationFrame(() => this.loop());
      });
  }
}

// ---------------------------------------------------------------------------
// Gesture classification helpers
// ---------------------------------------------------------------------------

// ─── Gesture state (hold smoothing) ──────────────────────────────────────────

let lastGesture: HandGesture = "unknown";
let gestureHoldCount = 0;

/**
 * Classify a hand gesture from 21 MediaPipe landmarks.
 *
 * Uses hold-frame smoothing: a gesture must persist for GESTURE_HOLD_FRAMES
 * consecutive frames before the classification changes.
 *
 * Landmark indices (MediaPipe convention):
 *   0  = WRIST
 *   4  = THUMB_TIP
 *   8  = INDEX_TIP
 *   12 = MIDDLE_TIP
 *   16 = RING_TIP
 *   20 = PINKY_TIP
 *   5  = INDEX_MCP  (knuckle)
 *   9  = MIDDLE_MCP
 *   13 = RING_MCP
 *   17 = PINKY_MCP
 */
function classifyGesture(landmarks: HandLandmark[]): HandGesture {
  if (landmarks.length < 21) return "unknown";

  const wrist = landmarks[0]!;
  const thumbTip = landmarks[4]!;
  const indexTip = landmarks[8]!;
  const middleTip = landmarks[12]!;
  const ringTip = landmarks[16]!;
  const pinkyTip = landmarks[20]!;

  const indexMcp = landmarks[5]!;
  const middleMcp = landmarks[9]!;
  const ringMcp = landmarks[13]!;
  const pinkyMcp = landmarks[17]!;

  const fingerExtended = (tip: HandLandmark, mcp: HandLandmark): boolean =>
    distance(tip, wrist) > distance(mcp, wrist) * 1.25;

  const indexExt = fingerExtended(indexTip, indexMcp);
  const middleExt = fingerExtended(middleTip, middleMcp);
  const ringExt = fingerExtended(ringTip, ringMcp);
  const pinkyExt = fingerExtended(pinkyTip, pinkyMcp);

  // Pinch: thumb and index tips are close
  const pinchDist = distance(thumbTip, indexTip);
  const wristMcpDist = distance(wrist, indexMcp);
  const pinching = pinchDist < PINCH_THRESHOLD * wristMcpDist * 10;

  let raw: HandGesture;
  if (pinching) {
    raw = "pinch";
  } else if (indexExt && !middleExt && !ringExt && !pinkyExt) {
    raw = "point";
  } else if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
    raw = "fist";
  } else if (indexExt && middleExt && ringExt && pinkyExt) {
    raw = "open";
  } else {
    raw = "unknown";
  }

  // Hold-frame smoothing: only switch after GESTURE_HOLD_FRAMES consistent frames
  if (raw === lastGesture) {
    gestureHoldCount = Math.min(gestureHoldCount + 1, GESTURE_HOLD_FRAMES);
  } else {
    gestureHoldCount = 1;
  }

  if (gestureHoldCount >= GESTURE_HOLD_FRAMES) {
    lastGesture = raw;
  }
  return lastGesture;
}

function distance(a: HandLandmark, b: HandLandmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
