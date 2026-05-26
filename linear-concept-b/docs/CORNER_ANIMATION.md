# Corner Animation — TRANSFORMED Phase

## Before (v1)
Corners snapped instantly to their transformed position when entering TRANSFORMED:
```
CONFIRM_TRANSFORM → Yes → TRANSFORMED (corners already at final position)
```

`drawCorners()` applied the transformation matrix directly with no interpolation.

## After (v2)
Corners smoothly slide from original grid position → transformed position over 1.5s with ease-out cubic.

**But v2 had two bugs:**
1. **Fill didn't animate** — `drawVirtualObject()` (the blue filled quadrilateral) used direct transformation with no interpolation. Only the outline in `drawCorners()` animated.
2. **Perpetual restart** — On completion, `animStartTime` was reset to `0` which is also the "not started" sentinel. Every animation loop completion (`animT >= 1`), `animStartTime` was set to `0`, and the next frame would trigger `animStartTime = Date.now()` again, restarting the animation infinitely. Each cycle was ~1.5s.

## Now (v3)
**Three fixes applied simultaneously:**

1. **Fill now animates** — `drawVirtualObject()` shares the same animation state via `getAnimationProgress(phase)` and interpolates corner positions from original → transformed using the same eased value.

2. **One-shot animation** — Completion sentinel changed from `0` to `-1`. On `animT >= 1`, `animStartTime = -1`. The start condition (`animStartTime === 0`) no longer matches, so the animation plays exactly once per entry to TRANSFORMED and never loops.

3. **Slower** — `ANIM_DURATION` increased from 1500ms → 10000ms.

## Architecture
- `getAnimationProgress(phase)` method encapsulates animation start/reset/completion logic. Called once in `draw()`, result passed to both `drawCorners()` and `drawVirtualObject()`.
- Animation only activates during `TRANSFORMED` phase. All other phases get `animT = 1` (fully transformed position, no interpolation).
- When `animT < 1`, positions are interpolated as `original + (transformed - original) * eased` where `eased = 1 - (1 - animT)³` (ease-out cubic).

## Files Changed
- `src/components/tabletopui.ts` — add `getAnimationProgress()`, extract animation state, use in both draw methods, fix sentinel, bump duration.

## Files Not Changed
- No changes to `main.ts` or `appstatemachine.ts`.
