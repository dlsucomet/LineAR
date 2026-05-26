# Configuration Reference

All tunable constants, grouped by source file.

## `src/main.ts`

### Camera

| Constant | Value | Description |
|---|---|---|
| `CAM_W` | 1280 | Camera capture width |
| `CAM_H` | 720 | Camera capture height |
| `CAM_FPS` | 30 | Camera capture framerate |
| `BOARD_HALF` | 10 | Half the grid board size |
| `STABLE_THRESHOLD` | 10 | Frames of stable detection before lock-in |
| `ABSENT_THRESHOLD` | 25 | Frames of no detection before clearing |
| `OPENCV_TIMEOUT` | 10000 | Timeout for OpenCV.js initialization (ms) |
| `FINGER_GUIDED_DEMO` | false | Enable demo mode (bypasses camera detection) |

### Corner Adjustment

| Constant | Value | Description |
|---|---|---|
| `CORNER_LOCK_DELAY_MS` | 3000 | Auto-lock after corner held stable on snap (ms) |
| `CORNER_COOLDOWN_MS` | 4000 | Cooldown before corner can be re-grabbed (ms) |
| `CORNER_MOVE_THRESHOLD_PX` | 10 | Finger movement threshold to reset snap timer (px) |
| `GRID_BOUNDS` | 15 | Maximum grid coordinate bound |

### Hand Tracking — Post-Processing

| Constant | Value | Description |
|---|---|---|
| `HAND_TIMEOUT_MS` | 400 | Grace period before falling back to mouse (ms) |
| `EPSILON` | 0.001 | Floating point epsilon for zero-crossing |
| `SMOOTH_ALPHA` | 0.5 | EMA smoothing factor for raw landmark coordinates |
| `DEADZONE_PX` | 8 | Minimum pixel movement to accept new position (both callbacks) |

## `src/core/demoplayer.ts`

### Interaction Thresholds

| Constant | Value | Description |
|---|---|---|
| `HOVER_DWELL_MS` | 400 | Dwell time to activate any button/prompt (ms) |
| `GRAB_RADIUS_PX` | 60 | Grab radius for corner circles (px) |
| `ARROW_GRAB_RADIUS_PX` | 70 | Grab radius for basis arrow tips (px) |
| `ARROW_SNAP_PX` | 25 | Snap distance to nearest grid intersection (px) |
| `ARROW_DWELL_PLACE_MS` | 600 | Dwell time on snap to place arrow (ms) |
| `ARROW_PLACED_LOCK_MS` | 3000 | Time after placement before arrow auto-locks (ms) |
| `ARROW_LOCK_COOLDOWN_MS` | 5000 | Cooldown after lock before arrow is re-grabbable (ms) |
| `CORNER_SNAP_RADIUS` | 1.5 | Grid snap radius for corner drag (grid units) |

### Phase Auto-Advance Timings

| Constant | Value | Description |
|---|---|---|
| `PHASE_TIMINGS.OBJECT_DETECTED` | 1500 | Auto-advance delay in demo mode (ms) |
| `PHASE_TIMINGS.SHOW_CORNERS` | 1000 | Auto-advance to basis vectors (ms) |
| `PHASE_TIMINGS.SHOW_BASIS_VECTORS` | 1500 | Demo phase timing (ms) |
| `PHASE_TIMINGS.TRANSFORMED` | 2500 | Demo phase timing (ms) |

Note: TRANSFORMED auto-advance is commented out — Continue button is the only way to advance.

### Presets

| Constant | Description |
|---|---|
| `VIRTUAL_OBJECT` | Pre-placed virtual object corners + center (demo mode) |
| `PRESET_MATRIX` | Shear matrix `[1, 0.5, 0.3, 1]` |
| `GHOST_ARROWS` | Dotted arrow targets derived from PRESET_MATRIX columns |

## `src/core/handtracker.ts`

| Setting | Value | Description |
|---|---|---|
| TF.js backend | WebGL (`@tensorflow/tfjs-backend-webgl`) | Primary inference backend; WASM also imported as fallback |
| `WEBGL_CPU_FORWARD` | `false` | Forced to false to prefer GPU execution path |
| Model type | `lite` | Faster/less accurate model |
| `maxHands` | 1 | Single hand tracking (configured in bootstrap/main.ts) |
| Gesture smoothing | 2 frames | Gesture must be consistent 2 frames before classification |
| `FRAME_SKIP` | 2 | Run inference every 2nd animation frame (halves inference calls) |

Smoothing in `main.ts` callbacks: 8px deadzone + EMA alpha=0.5 applied to index fingertip landmark.

## `src/components/tabletopui.ts`

### Grid

| Setting | Value | Description |
|---|---|---|
| `GRID_RANGE` | 10 | Grid extends from -10 to +10 in both axes |
| `TEXT_STRIP_H` | 40 | Height of instruction text strip at top (px) |

### Colors

| Variable | Value | Usage |
|---|---|---|
| `C.gridLine` | `rgba(100,160,220,0.55)` | Grid lines |
| `C.gridBg` | `#eef4fb` | Grid background |
| `C.e1` | `#d53a3a` | Basis vector e1 — X axis (red) |
| `C.e2` | `#3ad56b` | Basis vector e2 — Y axis (green) |
| `C.corner` | `#222` | Corner circle stroke |
| `C.cornerFill` | `#fff` | Corner circle fill |
| `C.matrixText` | `#111` | Matrix label text |
| `C.instrBlue` | `#1a3a6b` | Instruction text (default) |
| `C.instrCyan` | `#1a7a9a` | Instruction text (TRANSFORMED, POINTS_CALCULATED) |
| `C.instrOrange` | `#c05000` | Instruction text (OBJECT_DETECTED) |

### Canvas Buttons

Buttons are drawn on the canvas in fixed position (not pannable):

- **Yes / No** — shown during CONFIRM_TRANSFORM and CONFIRM_RESET, positioned to the right of instruction text in the text strip area
- **Continue** — shown during TRANSFORMED, same position as Yes/No
- **Done** — shown during POINTS_CALCULATED (after corners adjusted), same area

Button dimensions: ~64×28px (Yes/No), 100×28px (Continue). Dwell-activated at `HOVER_DWELL_MS`.

### Pan

| Setting | Value | Description |
|---|---|---|
| Available phases | TRANSFORMED only | Pan disabled during CONFIRM_TRANSFORM and CONFIRM_RESET |
| Pan origin | 0,0 at grid centre | Pan offsets apply from grid centre + canvas centre |
| Coordinate labels | `(orig) → (transformed)` | Shown for each corner during TRANSFORMED and CONFIRM_RESET |

## `src/core/cameratracker.ts`

| Setting | Value | Description |
|---|---|---|
| `DETECTION_INTERVAL` | Every 5th frame | Run OpenCV detection every N frames |
| `LOCKOUT_FRAMES` | 30 | Skip first N frames after start (camera warm-up) |

Camera runs continuously in all phases. Detection handler returns early during interactive phases (SHOW_BASIS_VECTORS, CONFIRM_TRANSFORM, TRANSFORMED, CONFIRM_RESET) — does NOT count absent frames or dispatch OBJECTS_CLEARED.
