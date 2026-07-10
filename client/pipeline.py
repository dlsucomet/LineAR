import os
import re
import json
import cv2
import numpy as np
from datetime import datetime
import config
from logger import log_message

def paper_tracking_daemon():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    max_w, max_h = 2000, 2400
    dst_pts = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
    last_state = False
    
    while config.running:
        if config.cap is None:
            continue
        ret, frame = config.cap.read()
        if not ret or frame is None:
            continue
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(ids) >= 4:
            ids = ids.flatten()
            corner_map = {int(ids[i]): corners[i][0] for i in range(len(ids))}

            # Calibration: detect surface markers 4-7 (independent of paper tracking)
            calib_ids = [4, 5, 6, 7]
            if all(k in corner_map for k in calib_ids):
                calib_src = np.array([corner_map[k][0] for k in calib_ids], dtype="float32")
                calib_dst = np.array([config.CALIB_MARKER_SCREEN_POSITIONS[k] for k in calib_ids], dtype="float32")
                calib_M, _ = cv2.findHomography(calib_src, calib_dst)
                with config.shared_frame_lock:
                    config.proj_calib_matrix = calib_M
                    config.proj_calibrated = True

            # Paper tracking: detect paper markers 0-3
            if all(k in corner_map for k in [0, 1, 2, 3]):
                src_pts = np.array([
                    corner_map[0][0], corner_map[1][0],
                    corner_map[3][0], corner_map[2][0]
                ], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                _, M_inv = cv2.invert(M)
                with config.shared_frame_lock:
                    config.tracking_matrix = M_inv
                    config.warped_document = cv2.warpPerspective(frame, M, (max_w, max_h), flags=cv2.INTER_CUBIC)
                    config.paper_detected = True
                if not last_state:
                    log_message("Tracking Lock Acquired: Target sheet anchors located.")
                    last_state = True
                continue

        with config.shared_frame_lock:
            if not config.debug_preview:
                config.paper_detected = False
        if last_state:
            log_message("Tracking Lock Lost: Target sheet missing or occluded.")
            last_state = False

def detect_content_bands(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_per_row = (binary > 0).sum(axis=1)
    threshold = w * 0.01

    in_content = False
    bands = []
    band_start = 0
    for row in range(h):
        has_content = ink_per_row[row] > threshold
        if has_content and not in_content:
            band_start = row
            in_content = True
        elif not has_content and in_content:
            band_end = row
            in_content = False
            if band_end - band_start > 10:
                band_region = binary[band_start:band_end, :]
                cols_with_ink = np.where(band_region.max(axis=0) > 0)[0]
                if len(cols_with_ink) > 0:
                    bands.append((band_start, band_end, cols_with_ink.min(), cols_with_ink.max()))
    if in_content:
        cols_with_ink = np.where(binary[band_start:, :].max(axis=0) > 0)[0]
        if len(cols_with_ink) > 0 and h - band_start > 10:
            bands.append((band_start, h, cols_with_ink.min(), cols_with_ink.max()))
    return bands


def _ocr_region(region_img, reader):
    blue = region_img[:, :, 0]
    best_tokens = []
    for bs, c in [(15, 2), (15, 4), (31, 2), (31, 4)]:
        try:
            thresh = cv2.adaptiveThreshold(blue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, c)
            processed = cv2.bitwise_not(thresh)
            results = reader.readtext(processed, allowlist='0123456789,-', paragraph=False)
            tokens = []
            for (bbox, text, confidence) in results:
                cleaned = text.replace(" ", "")
                for match in re.finditer(r'-?\d+', cleaned):
                    token = match.group()
                    if token and token != "-":
                        tokens.append(token)
            if len(tokens) > len(best_tokens):
                best_tokens = tokens
        except Exception:
            continue
    return best_tokens


def ocr_problem_data():
    with config.shared_frame_lock:
        if config.warped_document is None:
            return
        local_sheet = config.warped_document.copy()

    log_message("Reading problem data from paper using content band detection...")

    captures_dir = os.path.join(config.PARTICIPANT_DIR, "ocr_captures")
    os.makedirs(captures_dir, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S_%f")[:-3]

    bands = detect_content_bands(local_sheet)
    log_message(f"Detected {len(bands)} content bands")

    for i, (y0, y1, x0, x1) in enumerate(bands):
        log_message(f"  Band {i}: rows {y0}-{y1}, cols {x0}-{x1}")

    if len(bands) < 3:
        log_message("Problem OCR: too few content bands, will retry")
        try:
            cv2.imwrite(os.path.join(captures_dir, f"prob_failed_{stamp}.png"), local_sheet)
        except Exception:
            pass
        config.is_processing = False
        return

    content_bands = [b for b in bands if b[0] > 5 and b[1] < bands[-1][1] - 5]

    merge_gap = int(50 * local_sheet.shape[0] / 1200)
    merged = []
    current = content_bands[0]
    for band in content_bands[1:]:
        if band[0] - current[1] < merge_gap:
            current = (current[0], band[1], min(current[2], band[2]), max(current[3], band[3]))
        else:
            merged.append(current)
            current = band
    merged.append(current)

    merged = [m for m in merged if m[1] - m[0] > 15]
    log_message(f"Merged into {len(merged)} content regions")

    if len(merged) < 1:
        log_message("Problem OCR: no valid content regions after merge, will retry")
        config.is_processing = False
        return

    prob_y0, prob_y1, prob_x0, prob_x1 = merged[0]
    log_message(f"Problem region (merged): rows {prob_y0}-{prob_y1}, cols {prob_x0}-{prob_x1}")

    third_w = (prob_x1 - prob_x0) // 3
    thirds = [
        ("u", prob_x0, prob_y0, third_w, prob_y1 - prob_y0),
        ("w", prob_x0 + third_w, prob_y0, third_w, prob_y1 - prob_y0),
        ("target", prob_x0 + 2 * third_w, prob_y0, prob_x1 - prob_x0 - 2 * third_w, prob_y1 - prob_y0),
    ]

    results = {}
    for label, rx, ry, rw, rh in thirds:
        crop = local_sheet[ry:ry + rh, rx:rx + rw]
        if crop.size == 0:
            continue
        try:
            cv2.imwrite(os.path.join(captures_dir, f"prob_{label}_raw_{stamp}.png"), crop)
        except Exception:
            pass
        tokens = _ocr_region(crop, config.ocr_reader)
        results[label] = tokens
        log_message(f"Problem OCR [{label}] -> {tokens}")

    try:
        u_vals = results.get("u", [])
        w_vals = results.get("w", [])
        t_vals = results.get("target", [])

        if len(u_vals) >= 2:
            w_x = int(w_vals[0]) if len(w_vals) >= 2 else 0
            w_y = int(w_vals[1]) if len(w_vals) >= 2 else 1
            config.active_vectors = [
                {"x": int(u_vals[0]), "y": int(u_vals[1]), "label": "u"},
                {"x": w_x, "y": w_y, "label": "w"},
            ]
        if len(t_vals) >= 2:
            tx, ty = int(t_vals[0]), int(t_vals[1])
            config.target_vector = (tx, ty)
            config.active_vectors.append({"x": tx, "y": ty, "label": "L(v)"})

        config.expected_answers = {}

        data_ok = len(u_vals) >= 2
        if data_ok:
            config.problem_loaded = True
            log_message(f"Problem loaded: u={u_vals}, w={w_vals}, target={t_vals}")
        else:
            log_message(f"Problem OCR: vector data incomplete (u={u_vals}, w={w_vals}) — will retry")
    except Exception as e:
        log_message(f"PROBLEM OCR FAILURE: {str(e)}")

    config.is_processing = False

def background_ocr_pipeline():
    with config.shared_frame_lock:
        if config.warped_document is None:
            log_message("Verification Failed: No warped document data available yet.")
            config.is_processing = False
            return
        if not config.paper_detected:
            log_message("WARNING: Paper not currently in view — reusing last known warp.")
        local_sheet = config.warped_document.copy()

    captures_dir = os.path.join(config.PARTICIPANT_DIR, "ocr_captures")
    os.makedirs(captures_dir, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S_%f")[:-3]
    try:
        cv2.imwrite(os.path.join(captures_dir, f"warped_{config.current_step}_{stamp}.png"), local_sheet)
    except Exception:
        pass
        
    log_message(f"Starting OCR evaluation routine for calculation phase: {config.current_step}")
    recognized_data = {}
    try:
        regions = config.CROP_REGIONS.get(config.current_step, {})
        for region_name, box in regions.items():
            top, left, width, height = box["top"], box["left"], box["width"], box["height"]
            img_h, img_w = local_sheet.shape[:2]
            r_top = max(0, min(top, img_h - 1))
            r_left = max(0, min(left, img_w - 1))
            r_height = max(1, min(height, img_h - r_top))
            r_width = max(1, min(width, img_w - r_left))
            crop = local_sheet[r_top:r_top+r_height, r_left:r_left+r_width]
            
            blue_channel = crop[:, :, 0]
            thresh = cv2.adaptiveThreshold(blue_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
            processed_crop = cv2.bitwise_not(thresh)
            try:
                cv2.imwrite(os.path.join(captures_dir, f"{config.current_step}_{region_name}_{stamp}.png"), processed_crop)
            except Exception:
                pass
            ocr_results = config.ocr_reader.readtext(processed_crop, allowlist='0123456789-', paragraph=False)
            detected_tokens = []
            for (bbox, text, confidence) in ocr_results:
                cleaned = text.replace(" ", "")
                for match in re.finditer(r'-?\d+', cleaned):
                    token = match.group()
                    if token and token != "-":
                        y_center = bbox[0][1] + (bbox[2][1] - bbox[0][1]) / 2
                        detected_tokens.append((y_center, token))
            detected_tokens.sort(key=lambda item: item[0])
            recognized_data[region_name] = [t for (_, t) in detected_tokens]
            log_message(f"Parsed region '{region_name}' values -> {recognized_data[region_name]}")
        
        # Linear algebra step validation (dynamic)
        is_valid = False
        expected = config.expected_answers.get(config.current_step, {})
        all_empty = all(len(v) == 0 for v in recognized_data.values())
        if not all_empty and recognized_data == expected:
            is_valid = True
            if config.current_step == "thirdStep" and config.target_vector:
                tx, ty = config.target_vector
                config.active_vectors.append({"x": tx, "y": ty, "label": "L(v)"})

        if is_valid:
            config.green_count += 1
            config.feedback_state = "green"
            config.feedback_timer = 60
            idx = config.STEP_SEQUENCE.index(config.current_step)
            if idx < len(config.STEP_SEQUENCE) - 1:
                config.current_step = config.STEP_SEQUENCE[idx + 1]
                log_message(f"SUCCESS: Moving to step {config.current_step}")
            else:
                config.app_phase = "done"
                log_message("SUCCESS: Problem completed. Entering done phase.")
        else:
            config.red_count += 1
            config.feedback_state = "red"
            config.feedback_timer = 60
            log_message(f"REJECTED: Submission mismatch. Got: {list(recognized_data.values())}")
        
        with open(config.SESSION_PATH) as f:
            session_data = json.load(f)
        session_data["green_count"] = config.green_count
        session_data["red_count"] = config.red_count
        with open(config.SESSION_PATH, "w") as f:
            json.dump(session_data, f, indent=2)
            
    except Exception as e:
        log_message(f"CRITICAL OCR FAILURE: Exception thrown -> {str(e)}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    tracking_entry = f"[{timestamp}] OCR TRACK [{config.current_step}] regions -> {list(recognized_data.values())} paper_detected={config.paper_detected}\n"
    with open(config.LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(tracking_entry)

    config.is_processing = False

def transform_to_projection_space(w_x, w_y, center_rect):
    if config.tracking_matrix is None:
        return None
    src_point = np.array([[[w_x, w_y]]], dtype="float32")
    transformed = cv2.perspectiveTransform(src_point, config.tracking_matrix)
    cam_x = transformed[0][0][0]
    cam_y = transformed[0][0][1]
    if config.proj_calib_matrix is not None:
        cam_pt = np.array([[[cam_x, cam_y]]], dtype="float32")
        screen_pt = cv2.perspectiveTransform(cam_pt, config.proj_calib_matrix)
        calib_x, calib_y = screen_pt[0][0][0], screen_pt[0][0][1]
        p_x = center_rect.x + int((calib_x / config.WINDOW_WIDTH) * center_rect.width)
        p_y = center_rect.y + int((calib_y / config.WINDOW_HEIGHT) * center_rect.height)
        return int(p_x), int(p_y)
    p_x = center_rect.x + int((cam_x / 1280.0) * center_rect.width)
    p_y = center_rect.y + int((cam_y / 720.0) * center_rect.height)
    return int(p_x), int(p_y)