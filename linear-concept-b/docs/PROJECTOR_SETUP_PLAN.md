# Projector + Camera Setup Plan

## Goal
Project the webapp onto a surface via projector. A separate overhead camera captures the projection area. The system detects objects placed on the projection and tracks hands. The camera sees the projected grid — not a physical board.

## Risks

### 1. Camera sees the projected grid
OpenCV contour detection will find the grid rectangles instead of the physical object. This is the #1 blocker.

### 2. Hand model viewpoint
TFJS MediaPipe hands model (`lite`) was trained on front-facing camera data. An overhead (top-down) view of a hand on a surface is an unconventional angle — the model may fail to detect or track accurately.

### 3. Projector washout
Projected light on the user's hand and object changes how they appear in the camera feed, reducing contrast for both OpenCV and the hand model.

### 4. Camera/projector misalignment
Camera and projector have different FOVs, resolutions, and positions. Pixel coordinates don't align 1:1 without calibration.

### 5. Latency feedback loop
Camera → detect → render → project → camera. Each frame of latency accumulates.

---

## Plan: What to Code Now (No Hardware Needed)

### Phase 1 — Critical (Do before hardware arrives)

#### 1.1 Background Subtraction
**Why:** The only reliable way to ignore the projected grid is to detect only what's NEW in the frame.

**Concept:**
```
// At startup (before WAITING_FOR_OBJECT):
// 1. Capture a clean frame = camera sees only the projected grid, nothing else
// 2. Convert to grayscale, store as "background"

// Each detection frame:
// 1. Convert current frame to grayscale
// 2. absdiff(current, background) = difference image
// 3. threshold(difference, 30) = binary mask of changed pixels
// 4. findContours on the mask (not the raw frame)
// 5. Only changed regions become candidate objects
```

**New file:** `src/core/backgroundsubtractor.ts`
- `class BackgroundSubtractor`
- `captureBackground(video: HTMLVideoElement): void` — store reference frame
- `hasBackground(): boolean`
- `getForegroundMask(video: HTMLVideoElement): ImageData` — returns binary mask of changed pixels
- Configurable threshold value

**Modify:** `src/core/cameratracker.ts`
- Accept optional `BackgroundSubtractor`
- Before contour detection, if a background is captured, use the foreground mask to filter detection regions
- If no background is available, fall back to current raw-frame detection

**Modify:** `src/main.ts`
- During `bootstrap()`, after camera starts but before `WAITING_FOR_OBJECT`: prompt user to clear the projection area, then call `captureBackground`

### Phase 2 — Important (Do before hardware arrives)

#### 2.1 Configurable Coordinate Mapping
**Why:** The current `camToBoard` assumes camera and display share the same pixel space. A projector setup breaks this assumption.

**Concept:**
```
// Possible transform pipeline:
// Option A (default, works today):
//   camera pixel → direct scale → grid space
//
// Option B (projector mode):
//   camera pixel → homography → projector pixel → scale → grid space
```

**Modify:** `src/utils/helpers.ts`
- Wrap `camToBoard` in a function that checks which transform to use
- Add `setCameraTransform(transform: CameraTransform)` for mode switching
- `CameraTransform = "direct" | { type: "homography", matrix: number[] }`

**New file:** `src/core/calibration.ts` (stub for Phase 3)
- Define the calibration UI flow (new "CALIBRATING" phase)
- Store computed homography to localStorage for persistence

### Phase 3 — When Hardware is Available

#### 3.1 Calibration Procedure
1. Project 4+ markers at known grid positions
2. Camera detects them
3. Compute homography matrix (OpenCV `findHomography` or manual math)
4. Store it; swap `CameraTransform` from `"direct"` to `"homography"`

#### 3.2 Hand Tracking Evaluation
- Test TFJS MediaPipe hands with overhead camera angle
- If failure: evaluate alternative hand-detection approaches (e.g., simpler fingertip detection via depth, or infrared markers)
- The TFJS model path is preserved as-is — only empirical testing can validate it

#### 3.3 Exposure/Threshold Tuning
- Adjust background subtraction threshold to handle projector washout
- May need per-session calibration (auto-detect optimal threshold from histogram)

---

## Implementation Order

| Step | File(s) | Estimated LOC | Depends On |
|---|---|---|---|
| 1. BackgroundSubtractor class | `src/core/backgroundsubtractor.ts` | ~60 | Nothing |
| 2. Integrate into CameraTracker | `src/core/cameratracker.ts` | ~20 | Step 1 |
| 3. Capture background in bootstrap | `src/main.ts` | ~15 | Step 1, 2 |
| 4. Configurable coordinate chain | `src/utils/helpers.ts` | ~15 | Nothing |
| 5. Calibration module stub | `src/core/calibration.ts` | ~40 | Step 4 |
| 6. Homography compute + persist | (requires hardware) | — | Step 5 |
| 7. Hand tracking validation | (requires hardware) | — | Nothing |

---

## Files Changed Summary

```
NEW:  src/core/backgroundsubtractor.ts   (steps 1)
NEW:  src/core/calibration.ts             (step 5)
MOD:  src/core/cameratracker.ts           (step 2)
MOD:  src/main.ts                         (step 3)
MOD:  src/utils/helpers.ts                (step 4)
```
