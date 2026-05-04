// ─── src/core/handTracker.ts ──────────────────────────────────────────────────
// Wraps @tensorflow-models/hand-pose-detection to detect hands and classify
// basic gestures from a video element.

import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import "@tensorflow/tfjs-backend-webgl";
import type { DetectedHand, HandGesture, HandLandmark } from "../types/index.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Maximum number of hands to track simultaneously. */
const MAX_HANDS = 2;

/** Pinch distance threshold (normalised units). */
const PINCH_THRESHOLD = 0.07;

// ---------------------------------------------------------------------------
// HandTracker class
// ---------------------------------------------------------------------------

export class HandTracker {
  private detector: handPoseDetection.HandDetector | null = null;
  private isRunning = false;
  private animFrameId: number | null = null;
  private frameCallback: ((hands: DetectedHand[]) => void) | null = null;
  private videoEl: HTMLVideoElement | null = null;

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /** Initialise the TensorFlow model (downloads weights). */
  async init(): Promise<void> {
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig: handPoseDetection.MediaPipeHandsMediaPipeModelConfig = {
      runtime: "mediapipe",
      solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
      modelType: "full",
      maxHands: MAX_HANDS,
    };
    this.detector = await handPoseDetection.createDetector(model, detectorConfig);
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

    this.detector
      .estimateHands(this.videoEl, { flipHorizontal: false })
      .then((rawHands) => {
        const hands: DetectedHand[] = rawHands.map((h) => {
          const landmarks = h.keypoints3D?.map((kp) => ({
            x: kp.x,
            y: kp.y,
            z: kp.z ?? 0,
            name: kp.name,
          })) ?? h.keypoints.map((kp) => ({ x: kp.x, y: kp.y, z: 0, name: kp.name }));

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
      .catch(() => {
        // Silently skip frames that fail (e.g. video not ready)
        this.animFrameId = requestAnimationFrame(() => this.loop());
      });
  }
}

// ---------------------------------------------------------------------------
// Gesture classification helpers
// ---------------------------------------------------------------------------

/**
 * Classify a hand gesture from 21 MediaPipe landmarks.
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

  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];

  // Knuckle reference points
  const indexMcp = landmarks[5];
  const middleMcp = landmarks[9];
  const ringMcp = landmarks[13];
  const pinkyMcp = landmarks[17];

  const fingerExtended = (tip: HandLandmark, mcp: HandLandmark): boolean =>
    distance(tip, wrist) > distance(mcp, wrist) * 1.2;

  const indexExt = fingerExtended(indexTip, indexMcp);
  const middleExt = fingerExtended(middleTip, middleMcp);
  const ringExt = fingerExtended(ringTip, ringMcp);
  const pinkyExt = fingerExtended(pinkyTip, pinkyMcp);

  // Pinch: thumb and index tips are close
  const pinching = distance(thumbTip, indexTip) < PINCH_THRESHOLD * distance(wrist, indexMcp) * 10;

  if (pinching && !middleExt && !ringExt && !pinkyExt) return "pinch";
  if (indexExt && !middleExt && !ringExt && !pinkyExt) return "point";
  if (!indexExt && !middleExt && !ringExt && !pinkyExt) return "fist";
  if (indexExt && middleExt && ringExt && pinkyExt) return "open";

  return "unknown";
}

function distance(a: HandLandmark, b: HandLandmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
