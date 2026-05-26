# Interaction Guide

## Pointer Input

Two input sources, resolved each frame in the render loop (`main.ts`):

1. **Hand tracking** (camera + TF.js WebGL backend) — index fingertip mapped to canvas coordinates. Active when `isHandActive` is true (hand seen within last 400ms).
2. **Mouse / touch** — fallback via `ui.getMouseCanvasPos()`.

Hand tracking uses TF.js WebGL backend (`@tensorflow/tfjs-backend-webgl`) with `WEBGL_CPU_FORWARD=false`. Inference runs every 2nd animation frame (`FRAME_SKIP=2`). The `lite` model is used with `maxHands: 1`.

### Smoothing Pipeline

Raw landmark coordinates from TF.js pass through two filters before use:

1. **8px deadzone** — position only updates if finger moved ≥8px since last frame (filters sensor noise)
2. **EMA smoothing** (`alpha=0.5`) — `newPos = prevPos + 0.5 * delta` (dampens jitter)

Applied in both `bootstrap` and `setupDemoTracker` callbacks.

## Dwell Interaction

All buttons and interactions use dwell (hover-and-hold) instead of click:

- **HOVER_DWELL_MS** (400ms) — time to dwell on a button/prompt to activate
- Dwell progress resets when pointer leaves the target area

## Canvas Panning

Pan is available only during **TRANSFORMED** phase. Enabled via pointer-down drag anywhere outside the Continue button.

- Pan offsets stored in `TabletopUI.panX` / `panY`
- Grid center = `(canvas.width/2 + panX, TEXT_STRIP_H + gridHeight/2 + panY)`
- All grid-drawn content (grid lines, corners, basis arrows) respects pan offset
- Origin 0,0 is at grid centre
- Pan not available during CONFIRM_TRANSFORM, CONFIRM_RESET, or SHOW_BASIS_VECTORS to avoid competing with button/arrow interaction

### Corner Coordinate Labels

During TRANSFORMED and CONFIRM_RESET, each corner displays a label: `(origX,origY) → (transformedX,transformedY)` showing the original and transformed grid coordinates.

## Basis Vectors (SHOW_BASIS_VECTORS Phase)

Two arrow tips (e1, e2) represent the 2×2 transformation matrix columns:

| Arrow | Matrix column | Initial position |
|---|---|---|
| e1 (red) | `[mat[0], mat[2]]` | (1, 0) |
| e2 (green) | `[mat[1], mat[3]]` | (0, 1) |

### Arrow Interaction Sequence

```
1. GRAB   → pointer within ARROW_GRAB_RADIUS_PX (70px) of arrow tip
2. DRAG   → move pointer; tip snaps to nearest grid intersection within ARROW_SNAP_PX (25px)
3. DWELL  → hold still on intersection for ARROW_DWELL_PLACE_MS (600ms) → arrow is "placed"
4. LOCK   → after ARROW_PLACED_LOCK_MS (3000ms) → arrow auto-locks in position
5. COOLDOWN → after ARROW_LOCK_COOLDOWN_MS (5000ms) → arrow becomes re-grabbable
```

- While dragging, the dwell timer resets if the pointer leaves the snap zone
- Placed arrows can be re-grabbed after a short cooldown (800ms)
- When BOTH arrows are locked → `BASIS_ADJUSTED` dispatched → advances to CONFIRM_TRANSFORM
- Ghost arrow targets (dashed, semi-transparent) show the preset target positions

### Coordinate System

Arrow tip positions are computed in grid space then mapped to canvas:

```
canvasX = gridCenterX + mat[i] * scale
canvasY = gridCenterY - mat[j] * scale    (Y is flipped)
```

Where `scale = min(canvas.width, gridHeight) / 20` and grid range is -10..10.

## Corner Interaction (POINTS_CALCULATED Phase)

Four corner circles of the detected object. Drag to adjust, snap to nearest integer grid intersection. Auto-release after `CORNER_LOCK_DELAY_MS` (3s) of stable snap.

## Transformed Phase

- Continue button at top-right of instruction text (dwell to advance to CONFIRM_RESET) — **only** way to advance (no auto-transition)
- Pan available anywhere outside Continue button
- Corner labels show `(origX,origY) → (transformedX,transformedY)` for each corner
- Basis arrows show the applied matrix; matrix label displayed next to arrow origin

## Coordinate Mapping

| Function | Purpose | Location |
|---|---|---|
| `gridToCanvas(p, grid)` | Grid coords → canvas pixels | `tabletopui.ts` |
| `canvasToGrid(p, grid)` | Canvas pixels → grid coords | `tabletopui.ts` |

Both use: center = `grid center + pan`, scale = `min(grid.width, grid.height) / 20`, Y-flip.

## Detection Phase Guard

During interactive phases (SHOW_BASIS_VECTORS, CONFIRM_TRANSFORM, TRANSFORMED, CONFIRM_RESET), the camera detection handler returns early — it does NOT count absent frames or dispatch `OBJECTS_CLEARED`. This prevents flickering from OpenCV failing to find quads in later phases where the object is intentionally absent.
