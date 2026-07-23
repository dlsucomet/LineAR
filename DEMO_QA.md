# LineAR Demo Q&A

## 1. Project Overview

**Q: What is LineAR?**
A: LineAR is an augmented reality training system that teaches students how to compute linear transformations in linear algebra. It uses a projector-camera setup to overlay guidance onto a physical paper template and validates handwritten answers via OCR in real time.

**Q: What problem does it solve?**
A: Students struggle to understand how 2D linear transformations (2x2 matrices) map vectors. LineAR provides step-by-step guided practice with immediate feedback, bridging the gap between abstract matrix operations and concrete geometric intuition.

**Q: What is the target audience?**
A: University students taking introductory linear algebra courses.

**Q: What makes this different from existing tools?**
A: Most linear algebra tools are screen-only (e.g., GeoGebra). LineAR works on physical paper using AR projection, so students practice handwriting — the modality they'd use in exams — while still getting real-time digital feedback.

---

## 2. Technical Architecture

**Q: What is the overall system architecture?**
A: The system has six main modules: `config.py` (global state), `runner.py` (main event loop), `pipeline.py` (ArUco tracking + OCR), `ui.py` (Pygame rendering), `matrix_utils.py` (linear algebra), and `questionnaires.py` (post-task surveys). Communication happens through shared mutable state in `config.py`.

**Q: Why did you choose Python?**
A: Python has strong libraries for the core requirements: OpenCV for ArUco detection and image processing, PaddleOCR for text recognition, and Pygame for rapid UI prototyping. It also allowed fast iteration during development.

**Q: Why Pygame instead of a web-based UI?**
A: Pygame gives direct pixel-level control for real-time projection mapping and overlay rendering with low latency. A web stack would add complexity with browser rendering pipelines and WebSocket communication. For a projector-camera system, direct framebuffer control is critical.

**Q: What are the two operating modes and why?**
A: "With Highlights" shows blue AR boxes on the paper indicating where to write answers. "No Highlights" is the control condition with no visual guidance. This allows a between-subjects study comparing AR-assisted vs. unassisted problem solving.

**Q: How does the application state machine work?**
A: The `app_phase` variable cycles through: `start` -> `scan_qr` -> `running` -> `nasa_tlx` -> `ueq_s` -> `done`. Each phase has distinct rendering and input handling logic.

---

## 3. ArUco Marker Tracking

**Q: How does the system track the paper?**
A: Four ArUco markers (IDs 0-3, using the `DICT_4X4_50` dictionary) are printed at the corners of the paper. The camera detects all four simultaneously and computes a perspective transform to warp the camera frame into a flat, perspective-corrected document image (2200x3200 pixels).

**Q: Why ArUco markers instead of QR codes for tracking?**
A: ArUco markers are specifically designed for pose estimation and tracking — they're optimized for detection robustness, sub-pixel corner refinement, and fast detection. QR codes are designed for data encoding, not spatial tracking.

**Q: How do you handle jitter?**
A: The inverse tracking matrix is smoothed using an Exponential Moving Average (EMA) with alpha=0.3. This dampens sub-pixel corner fluctuations without introducing noticeable lag.

**Q: What happens when markers are temporarily lost?**
A: A 0.5-second dropout grace period tolerates brief single-frame disappearances (motion blur, glare). If markers return within that window, tracking continues uninterrupted. After 0.5 seconds, tracking is declared lost and the "Align ArUco Markers" message appears.

**Q: How many markers need to be visible?**
A: All four markers (IDs 0, 1, 2, 3) must be visible simultaneously to compute the perspective transform.

---

## 4. OCR and Answer Validation

**Q: What OCR engine do you use and why?**
A: PaddleOCR with the PaddlePaddle backend. It was chosen over EasyOCR (the original) for better accuracy on handwritten digits and faster inference.

**Q: How does the OCR pipeline work end-to-end?**
A: 1) The warped document is cropped to the current step's region. 2) Two preprocessing methods are applied (CLAHE + unsharp masking, and adaptive thresholding + morphological closing). 3) PaddleOCR's `predict()` API runs on both preprocessed versions. 4) Results are merged, deduplicated, and integer tokens extracted via regex. 5) Tokens are compared against expected answers from the QR code.

**Q: Why two preprocessing methods?**
A: Different preprocessing techniques work better for different handwriting styles and lighting conditions. Running both and merging results improves overall recognition robustness.

**Q: How are expected answers determined?**
A: They're encoded in the QR code on the paper. When the QR is scanned, the system parses the problem data (basis vectors, target vector, coefficients) and computes expected answers for each step. This means the system works with any problem, not just hardcoded ones.

**Q: How does the matching work — exact match or fuzzy?**
A: Subset matching. The system checks that ALL expected tokens appear somewhere in the OCR output. It's lenient about extra recognized numbers, which improves tolerance for OCR noise.

**Q: How does the content band detection work (no_highlights mode)?**
A: The system converts the cropped zone to grayscale, applies Otsu thresholding, sums ink pixels per row, finds rows exceeding a density threshold, groups contiguous rows into bands (minimum 15px height), and splits tall bands (>25px) by detecting horizontal gaps. Each band is then OCR'd separately.

**Q: What's the OCR triggering logic?**
A: Three conditions must be met: 1) ArUco markers must be continuously visible for 3+ seconds (stabilization), 2) at least 5 seconds must have passed since the last OCR completion (cooldown), and 3) the processing lock must be free. OCR runs in a daemon thread so the UI stays responsive.

---

## 5. Projection and AR Overlays

**Q: How do you map paper coordinates to screen coordinates?**
A: A two-step transform: 1) The perspective matrix maps paper coordinates to camera pixel coordinates, 2) Camera pixels are then scaled to the center panel's screen position using `center_rect.x + (cam_x / CAM_W) * center_rect.width`.

**Q: How do the green/red feedback overlays work?**
A: After OCR validation, semi-transparent green or red rectangles are drawn over the answer region on the center panel for 5 seconds. The regions are defined in paper coordinates and transformed to screen space the same way as the blue projection boxes.

**Q: What vectors are displayed on the Cartesian plane?**
A: The basis vectors u and v (as column vectors of the transformation matrix), their transformed versions L(u) and L(v), and the target vector w with its transformed output L(w). At the "complete" step, a parallelogram morphs from the unit square to the transformed shape.

---

## 6. Evaluation Methodology

**Q: Why NASA-TLX?**
A: NASA-TLX is the gold standard for subjective workload assessment in HCI research. It measures six dimensions: Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, and Frustration. This allows comparison of cognitive workload between the highlights and no-highlights conditions.

**Q: Why UEQ-S?**
A: UEQ-S (User Experience Questionnaire - Short) measures both pragmatic quality (usefulness, efficiency, clarity) and hedonic quality (excitement, novelty). It captures the overall user experience beyond just task performance, which is important for evaluating an AR educational tool.

**Q: What is the study design?**
A: Between-subjects design comparing "With Highlights" (AR guidance) vs. "No Highlights" (control). Participants solve linear transformation problems, complete NASA-TLX and UEQ-S questionnaires, and their correct/incorrect answer counts and task completion times are logged.

**Q: What data do you collect?**
A: Per-problem: green count (correct), red count (incorrect), duration, early exit flag. Per-session: NASA-TLX responses (6 dimensions), UEQ-S responses (8 items), problem history. Everything is saved to per-participant JSON files.

---

## 7. Design Decisions and Tradeoffs

**Q: Why a printed paper template instead of a tablet?**
A: Paper is the natural medium for math problem-solving in exams. Training on paper better transfers to exam conditions. It also avoids the complexity and cost of providing tablets.

**Q: Why not use a more modern AR framework like ARKit/ARCore?**
A: Those are designed for mobile devices with depth sensors. LineAR uses a fixed projector-camera setup, which is more appropriate for a classroom/lab setting. OpenCV + ArUco provides sufficient tracking quality at lower complexity.

**Q: Why subset matching instead of exact matching for OCR?**
A: Handwritten digit recognition is inherently noisy. Exact matching would produce excessive false negatives (marking correct answers as wrong). Subset matching is more forgiving — it checks that the required numbers are present, ignoring extra recognized artifacts.

**Q: Why use a QR code to encode the problem data instead of hardcoding problems?**
A: This makes the system extensible — new problems can be generated by creating new QR codes, no code changes needed. It also means each paper can have a unique problem, enabling multiple students to work simultaneously.

---

## 8. Limitations and Future Work

**Q: What are the current limitations?**
A: 1) OCR accuracy on handwritten digits is imperfect — heavily stylized or messy handwriting may fail. 2) The system currently only handles 2D linear transformations (2x2 matrices). 3) The paper template is fixed — answer positions are hardcoded in `PROJECTION_REGIONS`. 4) The tracking requires all 4 markers visible simultaneously. 5) PaddleOCR loading takes several seconds on startup.

**Q: How would you improve the system in the future?**
A: 1) Explore more robust handwriting recognition (e.g., a custom-trained CNN). 2) Extend to 3D transformations or general matrix operations. 3) Support dynamic paper layouts via more flexible content detection. 4) Add a teacher dashboard for monitoring multiple students. 5) Use a GPU-accelerated OCR engine for faster processing.

**Q: What happens if the student writes outside the expected region?**
A: In highlights mode, the system uses fixed crop regions, so answers outside those regions won't be read. In no_highlights mode, content band detection finds any ink within a larger zone, making it more forgiving of placement variations.

**Q: How does the system handle multiple numbers in a single answer box?**
A: The OCR pipeline extracts all integer tokens from a region. For steps expecting multiple numbers (e.g., a vector `[3, -2]`), it checks that both expected tokens appear in the recognized set via subset matching.

---

## 9. Technical Deep-Dive

**Q: Why use CLAHE for image preprocessing?**
A: CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local contrast in unevenly lit scenes — common with projector-camera setups where the projector may cast uneven illumination on the paper.

**Q: What is the EMA smoothing constant and why 0.3?**
A: Alpha=0.3 means 30% weight to the new observation and 70% to the historical value. This was empirically tuned to balance responsiveness (tracking paper movement) with stability (filtering out sub-pixel jitter). Higher alpha = more responsive but more jitter; lower alpha = smoother but laggier.

**Q: How does the lazy OCR engine initialization work?**
A: PaddleOCR is only loaded when the user first clicks "Start" (via `start_session()`), not at application launch. This avoids the multi-second startup penalty if the user is just exploring the UI. The engine is stored in `config.ocr_reader` and reused across OCR calls.

**Q: Why daemon threads for OCR and tracking?**
A: Daemon threads die automatically when the main process exits, preventing orphaned processes. They also don't block the main Pygame event loop, keeping the UI responsive during OCR processing.
