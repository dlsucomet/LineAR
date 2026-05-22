# Architecture

## File Structure

```
src/
├── main.ts                          App entry point, render loop, interaction logic
├── types/index.ts                   Shared type definitions
├── components/
│   └── tabletopui.ts                Canvas rendering, UI drawing (grid, arrows, buttons, corners)
├── core/
│   ├── appstatemachine.ts           Phase state machine (reducer pattern)
│   ├── cameratracker.ts             OpenCV.js object detection
│   ├── demoplayer.ts                Demo data, interaction thresholds, phase timings
│   └── handtracker.ts               MediaPipe hand pose detection (TF.js WASM backend)
├── camera/
│   └── cameramanager.ts             Camera stream management
└── shims/
    └── mediapipe-hands.ts           Stub for @mediapipe/hands (WASM not used by TF.js path)
```

## Phase State Machine

```
WAITING_FOR_OBJECT
  → (object detected, stable 10 frames)  → OBJECT_DETECTED
  → (CORNERS_LOCKED)                      → POINTS_CALCULATED

POINTS_CALCULATED
  → (object removed, OBJECTS_CLEARED)     → SHOW_CORNERS
  → (auto 1.5s, PHASE_ADVANCE)           → SHOW_BASIS_VECTORS (skip corner adjust)

SHOW_CORNERS
  → (auto 1s, PHASE_ADVANCE)              → SHOW_BASIS_VECTORS

SHOW_BASIS_VECTORS   ←──────────────────────┐
  → (both arrows locked → BASIS_ADJUSTED)   │
       → CONFIRM_TRANSFORM                  │
                                           │
CONFIRM_TRANSFORM                           │
  → Yes (CONFIRM_YES) → TRANSFORMED         │
  → No  (CONFIRM_NO)  → SHOW_BASIS_VECTORS ─┘

TRANSFORMED
  → Continue button dwell → TRANSFORMATION_DONE → CONFIRM_RESET

CONFIRM_RESET
  → Yes → WAITING_FOR_OBJECT (full reset)
  → No  → TRANSFORMED
```

## Key Actions (appstatemachine.ts)

| Action | Trigger | Effect |
|---|---|---|
| `OBJECTS_DETECTED` | Camera finds object | Advances from WAITING → OBJECT_DETECTED |
| `OBJECTS_CLEARED` | Camera loses object | Depending on phase: resets or advances |
| `CORNERS_LOCKED` | Corners computed from object | OBJECT_DETECTED → POINTS_CALCULATED |
| `BASIS_ADJUSTED` | Both basis vectors locked | SHOW_BASIS_VECTORS → CONFIRM_TRANSFORM |
| `CONFIRM_YES` | User dwells Yes button | CONFIRM_TRANSFORM → TRANSFORMED, or CONFIRM_RESET → restart |
| `CONFIRM_NO` | User dwells No button | Returns to previous phase |
| `TRANSFORMATION_DONE` | Continue button dwell | TRANSFORMED → CONFIRM_RESET |
| `PHASE_ADVANCE` | Auto-timer | SHOW_CORNERS → SHOW_BASIS_VECTORS or POINTS_CALCULATED → SHOW_BASIS_VECTORS |

## Pointer Hierarchy

1. Hand tracking (camera, index fingertip landmark 8) — when `isHandActive`
2. Mouse / touch — fallback via `ui.getMouseCanvasPos()`

`isHandActive = latestHandPosition != null && (Date.now() - lastHandSeenTime < HAND_TIMEOUT_MS)`
