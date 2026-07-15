import os
import re
import json
import time
import cv2
import numpy as np
from datetime import datetime
import config
from logger import log_message

from pyzbar.pyzbar import decode

def paper_tracking_daemon():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    max_w, max_h = 2000, 2400
    dst_pts = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
    last_state = False
    last_calib_state = False
    last_seen_ids = set()
    
    while config.running:
        if config.cap is None:
            continue
        ret, frame = config.cap.read()
        if not ret or frame is None:
            continue
        with config.shared_frame_lock:
            config.latest_frame = frame.copy()
        corners, ids, _ = detector.detectMarkers(frame)
        calib_ids = [4, 5, 6, 7]
        # Report which ArUco marker IDs are currently visible, only when the set changes
        current_ids = set(int(i) for i in ids.flatten()) if ids is not None else set()
        if current_ids != last_seen_ids:
            if current_ids:
                id_str = " ".join(str(i) for i in sorted(current_ids))
                log_message(f"ArUco markers visible: {id_str}")
            else:
                log_message("ArUco markers visible: none")
            last_seen_ids = current_ids

        if ids is not None and len(ids) >= 4:
            ids = ids.flatten()
            corner_map = {int(ids[i]): corners[i][0] for i in range(len(ids))}

            # Calibration: detect surface markers 4-7 (independent of paper tracking)
            if all(k in corner_map for k in calib_ids):
                calib_src = np.array([corner_map[k][0] for k in calib_ids], dtype="float32")
                calib_dst = np.array([config.CALIB_MARKER_SCREEN_POSITIONS[k] for k in calib_ids], dtype="float32")
                calib_M, _ = cv2.findHomography(calib_src, calib_dst)
                with config.shared_frame_lock:
                    config.proj_calib_matrix = calib_M
                    config.proj_calibrated = True
                if not last_calib_state:
                    log_message("Calibration Lock Acquired: ArUco markers 4, 5, 6, 7 detected.")
                    last_calib_state = True
            else:
                missing = [k for k in calib_ids if k not in corner_map]
                if last_calib_state:
                    log_message(f"Calibration Lock Lost: missing marker id(s) {missing}.")
                    last_calib_state = False

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
        if last_calib_state:
            log_message("Calibration Lock Lost: fewer than 4 ArUco markers visible.")
            last_calib_state = False

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

    log_message("Scanning document surface for configuration codes (QR or Barcode)...")

    try:
        # pyzbar scans for both 1D linear barcodes and 2D QR codes by default
        detected_codes = decode(local_sheet)
        
        if not detected_codes:
            log_message("Problem Code: No valid QR code or barcode detected, retrying.")
            config.is_processing = False
            return

        for code in detected_codes:
            raw_string = code.data.decode('utf-8').strip()
            log_message(f"Code discovered! Type: {code.type} | Content: {raw_string}")
            
            # Using regex allows us to grab numbers regardless of the packaging format.
            # Handles barcode strings: "1 2 3 4 5 6 7 8 9 10 11 12" 
            # Handles legacy QR strings: "1,2,3,4|5,6,7,8|9,10,11,12"
            vals = [int(n) for n in re.findall(r'-?\d+', raw_string)]
            
            if len(vals) >= 12:
                config.active_vectors = [
                    {"x": vals[0], "y": vals[1], "label": "u"},
                    {"x": vals[4], "y": vals[5], "label": "w"},
                ]
                
                tx, ty = vals[8], vals[9]
                config.target_vector = (tx, ty)
                config.active_vectors.append({"x": tx, "y": ty, "label": "L(v)"})
                
                config.expected_answers = {
                    "firstStepLeft": [str(vals[0]), str(vals[1])],    
                    "firstStepRight": [str(vals[2]), str(vals[3])],   
                    "secondStepLeft": [str(vals[4]), str(vals[5])],
                    "secondStepRight": [str(vals[6]), str(vals[7])],
                    "thirdStep": [str(vals[10]), str(vals[11])]        
                }
                
                config.problem_loaded = True
                log_message(f"Problem successfully initialized via {code.type}! Target: {config.target_vector}")
                break
            else:
                log_message(f"Data validation error: Found {len(vals)} numbers, expected 12.")
                
    except Exception as e:
        log_message(f"CRITICAL CODE INITIALIZATION FAILURE: {str(e)}")

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
            
            # Grayscale channel conversion to remove color lens rolling banding
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Cubic up-scale by 2.0x to handle small text dimensions safely
            resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            # Localized grid normalization via CLAHE to erase overhead lighting glare
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            enhanced = clahe.apply(resized)
            
            # Blur edge artifacts out of the compressed stream
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Binarize directly with Otsu thresholding for absolute print contrast
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            try:
                cv2.imwrite(os.path.join(captures_dir, f"{config.current_step}_{region_name}_{stamp}.png"), binary)
            except Exception:
                pass
                
            ocr_results = config.ocr_reader.readtext(binary, allowlist='0123456789-', paragraph=False)
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

def get_projector_box_points(box, center_rect):
    """
    Helper function to convert a bounding box configuration dict
    from 2000x2400 document coordinates into a contour format usable by cv2.polylines
    on your projection canvas screen space.
    """
    top, left, w, h = box["top"], box["left"], box["width"], box["height"]
    paper_corners = [(left, top), (left + w, top), (left + w, top + h), (left, top + h)]
    
    projector_pts = []
    for (px, py) in paper_corners:
        screen_pt = transform_to_projection_space(px, py, center_rect)
        if screen_pt is not None:
            projector_pts.append(screen_pt)
            
    if len(projector_pts) == 4:
        return np.array(projector_pts, dtype=np.int32).reshape((-1, 1, 2))
    return None

def track_pen_tip(frame, screen_w, screen_h):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if config.pen_debug_collect > 0:
        ch, cs, cv_val = hsv[360, 640]
        config.pen_debug_samples.append((int(ch), int(cs), int(cv_val)))
        config.pen_debug_collect -= 1
        if config.pen_debug_collect == 0:
            hs = [s[0] for s in config.pen_debug_samples]
            ss = [s[1] for s in config.pen_debug_samples]
            vs = [s[2] for s in config.pen_debug_samples]
            print(f"\n>>> HSV DEBUG SUMMARY ({len(config.pen_debug_samples)} frames at center) <<<")
            print(f">>>  H: min={min(hs)} max={max(hs)} avg={sum(hs)//len(hs)}")
            print(f">>>  S: min={min(ss)} max={max(ss)} avg={sum(ss)//len(ss)}")
            print(f">>>  V: min={min(vs)} max={max(vs)} avg={sum(vs)//len(vs)}")
            print(f">>> Current config range: lower={config.pen_hsv_lower} upper={config.pen_hsv_upper}\n")

    mask = cv2.inRange(hsv, config.pen_hsv_lower, config.pen_hsv_upper)
    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours and config.pen_calibrating:
        b, g, r = cv2.split(frame)
        diff_g = cv2.subtract(r, g)
        diff_b = cv2.subtract(r, b)
        pink_mask = cv2.bitwise_and(
            cv2.threshold(diff_g, 25, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(diff_b, 25, 255, cv2.THRESH_BINARY)[1]
        )
        pink_mask = cv2.erode(pink_mask, None, iterations=3)
        pink_mask = cv2.dilate(pink_mask, None, iterations=3)
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_cx, frame_cy = 640, 360
        best = None
        best_dist = float('inf')
        for c in pink_contours:
            area = cv2.contourArea(c)
            if area < 500 or area > 50000:
                continue
            M_c = cv2.moments(c)
            if M_c["m00"] == 0:
                continue
            c_cx = int(M_c["m10"] / M_c["m00"])
            c_cy = int(M_c["m01"] / M_c["m00"])
            dist = (c_cx - frame_cx) ** 2 + (c_cy - frame_cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = c
        if best is not None:
            contours = [best]

    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    if config.pen_calibrating:
        h, s, v = hsv[cy, cx]
        config.pen_calib_samples.append((int(h), int(s), int(v), cv2.contourArea(largest)))
        have = len(config.pen_calib_samples)
        print(f"[CALIB] {have}/30  H={int(h)} S={int(s)} V={int(v)}  area={cv2.contourArea(largest):.0f}")
        if have >= 30:
            hs = [s[0] for s in config.pen_calib_samples]
            ss = [s[1] for s in config.pen_calib_samples]
            vs = [s[2] for s in config.pen_calib_samples]
            avg_h = sum(hs) // len(hs)
            avg_s = sum(ss) // len(ss)
            avg_v = sum(vs) // len(vs)
            print(f"\n>>> CALIBRATION SAMPLES <<<")
            print(f">>>  Avg HSV: H={avg_h} S={avg_s} V={avg_v}")
            print(f">>>  H range: {min(hs)}-{max(hs)}  S range: {min(ss)}-{max(ss)}  V range: {min(vs)}-{max(vs)}")
            if max(hs) - min(hs) > 40:
                print(f">>>  WARNING: H range too wide ({min(hs)}-{max(hs)}). Samples may be inconsistent.")
                print(f">>>  Hold the pen steadier and calibrate again.")
                config.pen_calibrating = False
                config.pen_calib_samples = []
            else:
                h_low = max(0, min(hs) - 10)
                h_high = min(179, max(hs) + 10)
                s_low = max(0, min(ss) - 30)
                s_high = min(255, max(ss) + 30)
                v_low = max(0, min(vs) - 30)
                v_high = min(255, max(vs) + 30)
                h_low = min(h_low, h_high)
                s_low = min(s_low, s_high)
                v_low = min(v_low, v_high)
                print(f">>> pen_hsv_lower = ({h_low}, {s_low}, {v_low})")
                print(f">>> pen_hsv_upper = ({h_high}, {s_high}, {v_high})")
                print(f">>> Auto-saving to config.py ...")
                config.pen_hsv_lower = (h_low, s_low, v_low)
                config.pen_hsv_upper = (h_high, s_high, v_high)
                _save_pen_hsv_config(h_low, s_low, v_low, h_high, s_high, v_high)
                config.pen_calibrating = False
                config.pen_calib_samples = []
    sx = int(cx / 1280 * screen_w)
    sy = int(cy / 720 * screen_h)
    raw = (sx, sy)

    smoothed = config.pen_smooth_pos
    if smoothed is not None:
        dx = raw[0] - smoothed[0]
        dy = raw[1] - smoothed[1]
        dist = (dx * dx + dy * dy) ** 0.5
        max_jump = 80
        if dist > max_jump:
            ratio = max_jump / dist
            raw = (int(smoothed[0] + dx * ratio), int(smoothed[1] + dy * ratio))
            dist = max_jump

    buf = config.pen_median_buffer
    buf.append(raw)
    if len(buf) > 3:
        buf.pop(0)
    if len(buf) >= 3:
        xs = sorted(p[0] for p in buf)
        ys = sorted(p[1] for p in buf)
        cleaned = (xs[1], ys[1])
    else:
        cleaned = raw

    if smoothed is None:
        config.pen_smooth_pos = cleaned
        config.pen_median_buffer = buf
        return config.pen_smooth_pos

    sdx = cleaned[0] - smoothed[0]
    sdy = cleaned[1] - smoothed[1]
    velocity = (sdx * sdx + sdy * sdy) ** 0.5

    alpha_slow = 0.15
    alpha_fast = 0.5
    vel_low = 5.0
    vel_high = 40.0

    if velocity <= vel_low:
        alpha = alpha_slow
    elif velocity >= vel_high:
        alpha = alpha_fast
    else:
        t = (velocity - vel_low) / (vel_high - vel_low)
        alpha = alpha_slow + t * (alpha_fast - alpha_slow)

    config.pen_smooth_pos = (
        int(smoothed[0] * (1 - alpha) + cleaned[0] * alpha),
        int(smoothed[1] * (1 - alpha) + cleaned[1] * alpha),
    )
    config.pen_median_buffer = buf
    return config.pen_smooth_pos


def update_pen_state(screen_pos):
    now = time.time()
    click_cooldown = 0.3
    if screen_pos is not None:
        if not config.pen_track_active:
            config.pen_track_active = True
            config.pen_track_start = (*screen_pos, now)
            config.pen_has_clicked = False
        config.pen_track_last = screen_pos
        config.pen_position = screen_pos
        config.pen_visible = True
        if not config.pen_has_clicked:
            elapsed = now - config.pen_track_start[2]
            dx = screen_pos[0] - config.pen_track_start[0]
            dy = screen_pos[1] - config.pen_track_start[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if (elapsed >= 0.4 and dist < 10
                    and (now - config.pen_last_click_time) > click_cooldown):
                config.pen_click_queue.append(screen_pos)
                config.pen_last_click_time = now
                config.pen_has_clicked = True
    else:
        if config.pen_track_active and not config.pen_has_clicked:
            elapsed = now - config.pen_track_start[2]
            dx = config.pen_track_last[0] - config.pen_track_start[0]
            dy = config.pen_track_last[1] - config.pen_track_start[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if (elapsed < 0.5 and dist < 20
                    and (now - config.pen_last_click_time) > click_cooldown):
                config.pen_click_queue.append(config.pen_track_last)
                config.pen_last_click_time = now
        config.pen_track_active = False
        config.pen_position = None
        config.pen_visible = False
        config.pen_smooth_pos = None
        config.pen_median_buffer = []


def _save_pen_hsv_config(h_low, s_low, v_low, h_high, s_high, v_high):
    path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(path) as f:
        text = f.read()
    text = re.sub(
        r'pen_hsv_lower = \(\d+, \d+, \d+\)',
        f'pen_hsv_lower = ({h_low}, {s_low}, {v_low})',
        text
    )
    text = re.sub(
        r'pen_hsv_upper = \(\d+, \d+, \d+\)',
        f'pen_hsv_upper = ({h_high}, {s_high}, {v_high})',
        text
    )
    with open(path, "w") as f:
        f.write(text)
    print(f">>> Saved to config.py permanently.")