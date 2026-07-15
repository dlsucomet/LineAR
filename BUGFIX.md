# QR Scan Never Fires — Root Cause & Fixes

## Problem

`ocr_problem_data()` (pyzbar) is never called. Camera sees the QR code, but the program never reaches it.

## Root Causes

1. **OCR pipeline blocks QR scan** — In the main loop, the OCR pipeline check (answer-reading) runs before the QR scan check and grabs the `is_processing` lock first.

2. **`paper_stable_since` never accumulates 3 seconds** — The timer resets every frame because the perspective matrix coefficients (scaled to 2200x2600) change by more than `atol=1e-2` due to sub-pixel ArUco marker jitter. A 1080p camera won't fix this.

3. **`paper_stable_since` not reset on tracking loss** — When markers 0-3 disappear, `paper_detected` goes False but `paper_stable_since` keeps its stale value. The OCR trigger (which doesn't check `paper_detected`) fires on the stale timer, blocks the QR scan.

## Fixes (3 changes)

### 1. Reorder checks in `runner.py` — QR scan before OCR pipeline
Move the QR scan `if` block **above** the OCR pipeline `if` block so it gets first dibs on `is_processing`.

### 2. Replace matrix comparison in `pipeline.py:85-86`
```python
# Before (resets every frame due to jitter)
if last_matrix is None or not np.allclose(M_inv, last_matrix, atol=1e-2):
    config.paper_stable_since = time.time()

# After (stamps once on first detection)
if last_matrix is None:
    config.paper_stable_since = time.time()
```

### 3. Reset timer on tracking loss in `pipeline.py:103`
```python
config.paper_detected = False
config.paper_stable_since = time.time()  # add this line
```
