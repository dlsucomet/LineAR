# Sequential Basis Vectors with Hysteresis Locking

## Problem

1. **No sequential flow**: Both arrows (red e1, green e2) can be grabbed simultaneously, making it unclear which to move first.
2. **Lock never completes**: Microscopic hand movements (jitter) cause the finger to briefly leave the `ARROW_SNAP_PX` (25px) zone around a grid intersection. This resets the dwell timer, so the 600ms dwell-to-place never finishes. Even if placement somehow succeeds, the 800ms re-grab cooldown is shorter than the 3000ms lock timer, so re-grab fires before lock.
3. **Unclear lock state**: The locked indicator is a white outline, hard to distinguish from the snapped state.

## Solution

### 1. Sequential phases (`basisPhase`)

Add `ds.basisPhase: 'e1' | 'e2' | 'done'` to `DemoInteractionState`.

| Phase | Grabbable | Instruction | Advance when |
|-------|-----------|-------------|-------------|
| `e1` | Red (e1) only | "Move the red arrow" | e1 locks → phase = `e2` |
| `e2` | Green (e2) only | "Move the green arrow" | e2 locks → phase = `done`, proceed |
| `done` | Neither | — | `BASIS_ADJUSTED` dispatched |

### 2. Hysteresis snap (fixes locking)

Replace dwell-to-place with immediate snap + dual-radius hysteresis.

| Event | Radius | Behavior |
|-------|--------|----------|
| Finger enters snap zone | 25px (`ARROW_SNAP_PX`) | Arrow snaps to grid, `e1Placed = true`, lock timer starts |
| Finger jitters within zone | 25-70px | Arrow stays snapped (hysteresis), timer keeps running |
| Finger leaves release zone | 70px (`ARROW_GRAB_RADIUS_PX`) | Arrow unsnaps, `e1Placed = false`, timer cancelled |
| Lock timer expires | 3000ms (`ARROW_PLACED_LOCK_MS`) | Arrow auto-locks (via existing section-1 logic) |

### 3. Remove premature re-grab

Delete the "re-grab if placed but not locked" code path. Once placed, the arrow stays placed until either:
- Lock timer completes → locked (then cooldown auto-unlocks after 5s for adjustment)
- Finger releases (moves >70px from intersection) → unsnaps

The existing cooldown system (locked → auto-unlock after `ARROW_LOCK_COOLDOWN_MS` = 5s) remains unchanged for adjustments.

### 4. Blue locked indicator

In `tabletopui.ts`, change `drawArrowDragFeedback` locked stroke from `"#fff"` to `C.cursorStroke` (`"#3a7bd5"` blue).

## Files changed

| File | What |
|------|------|
| `src/main.ts` | `basisPhase` state; hysteresis snap in e1/e2 drag; phase-gated grab; removed premature re-grab; dynamic instruction text |
| `src/components/tabletopui.ts` | Blue locked indicator |

## Variables reused

No new state variables needed — repurpose existing ones:
- `e1DwellIntersection` → snapped grid point (hysteresis anchor)
- `e1DwellStart` → lock timer start timestamp
- `e1Placed` / `e1PlaceTime` → set on snap, consumed by existing auto-lock
- `e1Locked` / `e1LockTime` → existing auto-lock + cooldown
