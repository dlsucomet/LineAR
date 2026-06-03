# ArUco Changes — 2024-06-03

Two changesets applied to switch from js-aruco (5×5 projected markers) to
OpenCV.js built-in ArUco (4×4 physical printed markers) with automatic
spatial grid-position assignment.

---

## Changeset 1: OpenCV.js ArUco 4×4 detection

**Goal:** Use physical 4×4 printed markers instead of projecting 5×5 markers.

### Files modified

| File | What changed |
|---|---|
| `index.html:66-72` | Commented out `aruco-marker`, `cv.js`, `aruco.js` script tags |
| `src/core/calibration.ts:3-8` | Removed `declare const AR` / `declare const CV` |
| `src/core/calibration.ts:60-138` | Rewrote `detectProjectedMarkers()` — uses `cv.aruco_ArucoDetector` with `getPredefinedDictionary`, tries DICT_4X4_50→100→250→1000 |
| `src/core/calibration.ts:170-184` | Removed deprecated OpenCV ArUco stub |
| `src/core/cameratracker.ts:9-14` | Removed `declare const AR: any; declare const CV: any` |
| `src/core/cameratracker.ts:78` | Added `private arucoDetector: any = null` (cached detector instance) |
| `src/core/cameratracker.ts:229-262` | Replaced js-aruco masking block with OpenCV's `aruco_ArucoDetector` |
| `src/components/tabletopui.ts:306-307` | Commented out `drawCalibrationMarkers(grid, state.phase)` |

### Revert

```
# index.html: uncomment lines 66-72, remove the --> on line 72
# calibration.ts: restore old AR/CV declarations + js-aruco detectProjectedMarkers
# camerateracker.ts: restore old AR/CV declarations + js-aruco masking block
# tabletopui.ts: uncomment line 307
```

---

## Changeset 2: Spatial grid-position assignment (no hardcoded ID→grid)

**Goal:** Auto-detect which marker belongs to which corner by sorting them
in camera space, instead of hardcoding `{ id: 1 → grid(-9,9) }`.

### Files modified

| File | What changed |
|---|---|
| `src/core/calibration.ts:64-70` | Added `gridPos: Point2D` to `PhysicalMarker` interface (required) |
| `src/core/calibration.ts:117` | Placeholder `gridPos: { x: 0, y: 0 }` when pushing candidates |
| `src/core/calibration.ts:122-123` | Call `assignGridPositions(candidates)` after finding ≥4 markers |
| `src/core/calibration.ts:140-153` | New function `assignGridPositions()` — sorts 4 markers by Y (top/bottom) then X (left/right), assigns grid positions |
| `src/core/calibration.ts:155-185` | Rewrote `computeCalibrationFromMarkers()` — uses `m.gridPos` from spatial sort instead of `gridMap.get(m.id)` |
| `src/core/calibration.ts` | Removed `const gridMap = new Map(...)` (no longer needed) |

### Sorting logic

```
1. Sort all 4 markers by center.y (ascending)
2. Top 2 → back row (gridY = 9), bottom 2 → front row (gridY = -9)
3. Within each row, sort by center.x (ascending)
   - Left → gridX = -9
   - Right → gridX = 9
```

### Edge cases

- Works best when camera is overhead at an angle (not rotated 90°+)
- If the workspace is rotated relative to camera view, the Y-sort may
  mis-assign rows. The homography will still be valid as long as the 4
  corners are correctly identified.

### Revert

```
# calibration.ts: remove assignGridPositions + gridPos field
# calibration.ts: restore old computeCalibrationFromMarkers with gridMap
# calibration.ts: restore ARUCO_MARKERS as the source of truth for grid positions
```

---

## Changeset 3: Coordinate flips for visual alignment

**Goal:** Object positions rendered on the grid should align with the
flipped camera background (flipH/flipV), so the visual display is
consistent for the user.

**Problem:** The camera background is flipped (H+V) to look natural from the
user's perspective at the table, and hand landmarks are separately flipped in
`toMirroredCanvas`. But object/marker coordinates were fed raw into the
homography, producing a 180° visual mismatch (top-right object → bottom-left on
grid).

**Fix:** Apply flipH/flipV to the coordinates *before* they enter the
homography, in both calibration and runtime paths.

### Files modified

| File | Before | After |
|---|---|---|
| `src/utils/coordinatemapper.ts:9-10` | (no flip fields) | `private flipH = false; private flipV = false;` + `setFlips(h, v)` method |
| `src/utils/coordinatemapper.ts:37-52` | `cameraToGrid` uses `pixel.x, pixel.y` directly | `cameraToGrid` flips input via `fx = flipH ? camW - pixel.x : pixel.x`, same for fy, then uses `fx,fy` |
| `src/core/calibration.ts:161-178` | `computeCalibrationFromMarkers(markers)` | `computeCalibrationFromMarkers(markers, camW?, camH?, flipH?, flipV?)` — flips marker centers before homography |
| `src/main.ts:154-156` | `ui.setFlipH(flipH); ui.setFlipV(flipV);` | `+ coordMapper.setFlips(flipH, flipV);` |
| `src/main.ts:530-535` | `runCalibrationSequence(…, onStatus)` | `runCalibrationSequence(…, onStatus, flipH?, flipV?)` |
| `src/main.ts:549` | `computeCalibrationFromMarkers(markers)` | `computeCalibrationFromMarkers(markers, res.width, res.height, flipH, flipV)` |
| `src/main.ts:337-341` | `runCalibrationSequence(cameraTracker, coordMapper, 10, onStatus)` | `+ flipH, flipV` as extra args |

### Coordinate flow after fix

```
Raw camera pixel ─► apply flipH/flipV ─► homography ─► grid coords ─► gridToCanvas ─► canvas pixel
                                           ↑
                                   (saved matrix was
                                    computed from
                                    flipped marker coords)
```

All three visual paths now use the same flips:
1. Camera background — flipped via `drawCameraBackground`
2. Hand landmarks — flipped via `toMirroredCanvas`
3. Object/marker coords — flipped via `cameraToGrid` / `computeCalibrationFromMarkers`

### Revert

```
# coordinatremapper.ts: remove flipH, flipV fields, setFlips, and flip logic from cameraToGrid
# calibration.ts: remove camW, camH, flipH, flipV params from computeCalibrationFromMarkers
# main.ts: remove coordMapper.setFlips() call
# main.ts: remove flipH, flipV params from runCalibrationSequence and its call site
```

Clearing the saved homography (`localStorage.clear()` + reload) is **required**
after revert, since the matrix was saved in the flipped coordinate space.
```
