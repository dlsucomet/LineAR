# Detection Threshold Tuning

## Problem

The contour-based object detector finds quadrilateral shapes in the environment (table grain, shadows, furniture edges, camera noise) and reports false positive detections before the user places any physical object.

## Root causes

1. **Background capture too short**: 10 frames (0.33s at 30fps) is insufficient for the camera auto-exposure to stabilize and the background model to converge.
2. **Subtraction threshold too low**: The pixel-difference threshold (30/255) is triggered by minor fluctuations in lighting, auto-exposure, and sensor noise.
3. **Minimum contour area too small**: 800 px² in a 1920×1080 frame is ~0.04% — small surface defects and noise edges exceed this.
4. **Detection consensus too lenient**: Only 4 out of 10 frames need to show a quad for the system to declare an object "present".

## Changes

| File | Constant | Before | After | Effect |
|------|----------|--------|-------|--------|
| `src/core/cameratracker.ts` | `MIN_CONTOUR_AREA` | 800 | 2500 | Filters small noise shapes; a 50×50px area is now the floor |
| `src/core/cameratracker.ts` | `BG_CAPTURE_FRAMES` | 10 | 30 | ~1 second background capture for camera to stabilize |
| `src/core/cameratracker.ts` | threshold in `cv.threshold(diff, fgMask, 30, ...)` | 30 | 45 | Less sensitive to minor pixel fluctuations |
| `src/core/cameratracker.ts` | `MIN_SIZE_FRACTION` | 0.05 | 0.08 | Object must be ≥8% of frame width/height |
| `src/main.ts` | `DETECTION_WINDOW_THRESHOLD` | 4/10 | 6/10 | Need 60% frame consensus instead of 40% |

## Verification

- A real object (business card, paper, phone) is easily >2500 px² and >8% of frame at arm's length
- Background capture over 30 frames gives a more stable average for subtraction
- Higher subtraction threshold ignores noise while still detecting real foreground changes
- Higher consensus window reduces fluke detections from single-frame noise
