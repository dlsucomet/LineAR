# Camera Flip Configuration

## Summary

The camera feed and hand tracking coordinates can be independently flipped
horizontally and/or vertically. This compensates for different camera mounting
orientations (e.g. camera mounted upside-down, mirror setups, etc.).

## Default values

The default flip state is read from `public/mirror-config.txt`:

```
# flipH,flipV  (1=enable, 0=disable)
1,1
```

| Value   | Horizontal flip (flipH) | Vertical flip (flipV) |
|---------|------------------------|----------------------|
| `1,1`   | Enabled                | Enabled              |
| `0,1`   | Disabled               | Enabled              |
| `1,0`   | Enabled                | Disabled             |
| `0,0`   | Disabled               | Disabled             |

Both flips are **enabled by default** (`1,1`) — the camera image is mirrored
on both axes so that moving your hand left on the table moves the cursor left
on the projection.

## How it works

### Config loading (`src/main.ts:38-49`)

```ts
async function loadMirrorConfig() {
  try {
    const res = await fetch("/mirror-config.txt");
    const lines = (await res.text()).split("\n")
      .filter(l => l.trim() && !l.trim().startsWith("#"));
    const parts = lines[0]?.trim().split(",") || [];
    flipH = parts[0] === "1";
    flipV = parts[1] === "1";
  } catch {
    flipH = false; flipV = false;
  }
}
loadMirrorConfig();
```

- Fetches `mirror-config.txt` at module load time
- Skips the `#` comment line, reads only the data line
- Sets module-level `flipH`, `flipV` booleans
- Falls back to `false, false` if the file is missing or unreadable
- The render loop syncs these values to the UI each frame

### Hand tracking coordinates (`src/main.ts:631-634`)

```ts
const toMirroredCanvas = (lm: HandLandmark): Point2D => ({
  x: (flipH ? vw - lm.x : lm.x) * uniformScale + offsetX,
  y: (flipV ? vh - lm.y : lm.y) * uniformScale + offsetY,
});
```

- MediaPipe landmark coordinates (`lm.x`, `lm.y`) are in video pixel space
- When `flipH` is `true`: `vw - lm.x` mirrors horizontally
- When `flipV` is `true`: `vh - lm.y` mirrors vertically
- Then scaled and offset to canvas coordinates (letterboxed)

### Camera background draw (`src/components/tabletopui.ts:263-268`)

```ts
ctx.save();
ctx.translate(W / 2, H / 2);
ctx.scale(this.flipH ? -1 : 1, this.flipV ? -1 : 1);
ctx.translate(-W / 2, -H / 2);
ctx.drawImage(this.videoEl, dx, dy, vw * scale, vh * scale);
ctx.restore();
```

- Canvas context is scaled by `-1` on each axis independently (mirror)
- `flipH = -1` mirrors horizontally, `flipV = -1` mirrors vertically
- The video is drawn at the same scale used for hand landmarks (`Math.min`)
- Both camera display and hand coordinates use identical flip logic

### Runtime toggles (`src/main.ts:181-191`)

| Key | Effect |
|-----|--------|
| **F** | Toggle `flipH` immediately |
| **V** | Toggle `flipV` immediately |

No config file reload needed — the booleans update on the next frame.

## Effect on cursor movement

| Configuration | Hand moves left → cursor moves |
|--------------|-------------------------------|
| `flipH=0, flipV=0` | Right (no mirror — raw camera) |
| `flipH=1, flipV=0` | Left (horizontally mirrored) |
| `flipH=0, flipV=1` | Left (vertically mirrored — combined) |
| `flipH=1, flipV=1` | Left (both axes flipped) |

The default (`1,1`) makes cursor and hand move in the same direction on the
projected surface.

## Files involved

| File | Role |
|------|------|
| `public/mirror-config.txt` | Persistent default values (comment + data line) |
| `src/main.ts:38-49` | `loadMirrorConfig()` — read and parse config |
| `src/main.ts:181-191` | F/V key handlers — runtime toggle |
| `src/main.ts:631-634` | `toMirroredCanvas()` — apply flips to hand coords |
| `src/components/tabletopui.ts:263-268` | `drawCameraBackground()` — apply flips to video feed |
| `src/main.ts:233-234` | Render loop — sync flip flags each frame |
