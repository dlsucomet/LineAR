import os
import re
import json
import time
import cv2
import numpy as np
from collections import Counter
from datetime import datetime
import config
from logger import log_message

from pyzbar.pyzbar import decode


def paper_tracking_daemon():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.adaptiveThreshWinSizeMin = 7
    aruco_params.adaptiveThreshWinSizeMax = 53
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementWinSize = 10
    aruco_params.cornerRefinementMaxIterations = 50
    aruco_params.minDistanceToBorder = 5
    aruco_params.errorCorrectionRate = 0.8
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    max_w, max_h = 2200, 2600  # marker-corner mapping — DO NOT change, existing crop regions are calibrated against this
    pad = 100
    dst_pts = np.array([[pad, pad], [max_w - 1 - pad, pad], [max_w - 1 - pad, max_h - 1 - pad], [pad, max_h - 1 - pad]], dtype="float32")
    # The output canvas can be larger than the marker-mapped area without affecting where markers/existing
    # regions land — it just reveals more of whatever the perspective transform maps beyond the old edges.
    # The QR/barcode sits below the bottom markers and was getting clipped by the old max_h boundary.
    canvas_w, canvas_h = max_w, max_h + 600
    last_state = False
    last_seen_ids = set()
    last_matrix = None
    last_src_pts = None
    smoothed_tracking = None
    ema_alpha = 0.3
    last_capture_time = 0
    STABILITY_PIXEL_TOLERANCE = 3.0  # max per-corner movement (px) to still count as "stable"
    last_full_detect_time = 0.0
    MARKER_DROPOUT_GRACE = 0.5  # seconds to tolerate a brief marker dropout before declaring lock lost
    
    while config.running:
        if config.cap is None:
            continue
        ret, frame = config.cap.read()
        if not ret or frame is None:
            continue
        # flipped
        frame = cv2.flip(frame, -1)
        with config.shared_frame_lock:
            config.latest_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        corners, ids, _ = detector.detectMarkers(enhanced_bgr)
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
            corner_map = {int(ids[i]): np.mean(corners[i][0], axis=0) for i in range(len(ids))}

            # Paper tracking: detect paper markers 0-3
            if all(k in corner_map for k in [0, 1, 2, 3]):
                src_pts = np.array([
                    corner_map[0], corner_map[1],
                    corner_map[3], corner_map[2]
                ], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                _, M_inv = cv2.invert(M)
                if last_src_pts is None:
                    config.paper_stable_since = time.time()
                else:
                    max_corner_shift = np.max(np.linalg.norm(src_pts - last_src_pts, axis=1))
                    if max_corner_shift > STABILITY_PIXEL_TOLERANCE:
                        config.paper_stable_since = time.time()
                last_src_pts = src_pts.copy()
                last_matrix = M_inv.copy()
                last_full_detect_time = time.time()
                if smoothed_tracking is None:
                    smoothed_tracking = M_inv.copy()
                else:
                    smoothed_tracking = ema_alpha * M_inv + (1 - ema_alpha) * smoothed_tracking
                with config.shared_frame_lock:
                    config.tracking_matrix = smoothed_tracking
                    config.warped_document = cv2.warpPerspective(frame, M, (canvas_w, canvas_h), flags=cv2.INTER_CUBIC)
                    config.paper_detected = True
                if not last_state:
                    log_message("Tracking Lock Acquired: Target sheet anchors located.")
                    last_state = True

                now_capture = time.time()
                is_settled = now_capture - config.paper_stable_since >= 1.0  # matrix hasn't moved in the last 1s
                if config.PARTICIPANT_DIR and is_settled and now_capture - last_capture_time >= 3.0:
                    last_capture_time = now_capture
                    captures_dir = os.path.join(config.PARTICIPANT_DIR, "ocr_captures")
                    os.makedirs(captures_dir, exist_ok=True)
                    stamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                    try:
                        crop_img = config.warped_document
                        regions = config.CROP_REGIONS.get(config.current_step, {})
                        if regions:
                            imgs = []
                            for rname, box in regions.items():
                                ih, iw = crop_img.shape[:2]
                                r_top = max(0, min(box["top"], ih - 1))
                                r_left = max(0, min(box["left"], iw - 1))
                                r_height = max(1, min(box["height"], ih - r_top))
                                r_width = max(1, min(box["width"], iw - r_left))
                                imgs.append(crop_img[r_top:r_top+r_height, r_left:r_left+r_width])
                            crop_img = np.vstack(imgs) if len(imgs) > 1 else imgs[0]
                        cv2.imwrite(
                            os.path.join(captures_dir, f"p{config.problem_number}_track_{config.current_step}_{stamp}.png"),
                            crop_img,
                        )
                    except Exception:
                        pass
                continue

        now = time.time()
        if last_state and last_full_detect_time and (now - last_full_detect_time) < MARKER_DROPOUT_GRACE:
            # Brief single-frame marker dropout (motion blur, glare, angle) — keep the existing
            # lock, tracking matrix, and stability timer intact rather than resetting everything.
            pass
        else:
            with config.shared_frame_lock:
                if not config.debug_preview:
                    config.frozen_tracking_matrix = config.tracking_matrix
                    config.paper_detected = False
            last_matrix = None
            last_src_pts = None
            last_capture_time = 0  # reset so the next lock starts its own capture cadence
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


def scan_qr_from_camera():
    with config.shared_frame_lock:
        if config.latest_frame is None:
            config.is_processing = False
            return
        local_frame = config.latest_frame.copy()

    log_message("Scanning raw camera frame for QR code...")

    try:
        data = None
        code_type = None

        detector = cv2.QRCodeDetector()
        decoded, _, _ = detector.detectAndDecode(local_frame)
        if decoded:
            data = decoded.strip()
            code_type = "QRCODE"

        if not data:
            codes = decode(local_frame)
            if codes:
                data = codes[0].data.decode('utf-8').strip()
                code_type = codes[0].type

        if not data:
            log_message("QR Code: No code detected in camera frame, retrying.")
            config.is_processing = False
            return

        log_message(f"Code discovered! Type: {code_type} | Content: {data}")

        sections = data.rstrip('|').split('|')
        if len(sections) != 6:
            log_message(f"Invalid QR format: expected 6 pipe-delimited sections, got {len(sections)}.")
            config.is_processing = False
            return

        s1 = sections[0].split(';')
        g1, v1 = [int(x) for x in s1[0].split(',')]
        og1, ov1 = [int(x) for x in s1[1].split(',')]

        s2 = sections[1].split(';')
        g2, v2 = [int(x) for x in s2[0].split(',')]
        og2, ov2 = [int(x) for x in s2[1].split(',')]

        s3 = sections[2].split(';')
        f1, n1 = [int(x) for x in s3[0].split(',')]
        of1, on1 = [int(x) for x in s3[1].split(',')]

        c1, c2 = [int(x) for x in sections[3].split(',')]

        s6 = sections[5].split(';')
        c1og1, c1ov1 = [int(x) for x in s6[0].split(',')]
        c2og2, c2ov2 = [int(x) for x in s6[1].split(',')]

        config.qr_data = {
            'step1_linearCombination': {
                'c1': c1, 'c2': c2,
                'v1': [g1, v1], 'v2': [g2, v2],
                'v_target': [f1, n1]
            },
            'step2_applyTransformation': {
                'c1': c1, 'c2': c2,
                'og1_ov1': [og1, ov1], 'og2_ov2': [og2, ov2]
            },
            'step3_scalarMultiplication': {
                'scaled_vector1': [c1og1, c1ov1],
                'scaled_vector2': [c2og2, c2ov2]
            },
            'complete': {
                'final_output': [of1, on1],
                'v_target': [f1, n1]
            }
        }

        config.active_vectors = [
            {"x": g1, "y": v1, "label": "u", "show_at": "firstStep"},
            {"x": g2, "y": v2, "label": "v", "show_at": "firstStep"},
            {"x": f1, "y": n1, "label": "w", "type": "point", "show_at": "firstStep"},
            {"x": of1, "y": on1, "label": "L(w)", "type": "point", "show_at": "complete"},
        ]
        config.target_vector = (of1, on1)

        config.expected_answers = {
            "firstStepOne":    {"one": [str(f1), str(n1)]},
            "firstStepTwo":    {"one": [str(g1), str(v1)]},
            "firstStepThree":  {"one": [str(g2), str(v2)]},
            "firstStepFour":   {"one": [str(c1), str(c2)]},
            "secondStepLeft":  {"one": [str(c1), str(og1), str(ov1)]},
            "secondStepRight": {"one": [str(c2), str(og2), str(ov2)]},
            "thirdStepLeft":   {"one": [str(c1og1), str(c1ov1)]},
            "thirdStepRight":  {"one": [str(c2og2), str(c2ov2)]},
            "fourthStep":      {"one": [str(of1), str(on1)]},
        }

        log_message(f"QR parsed — u=({g1},{v1}) L(u)=({og1},{ov1}) | v=({g2},{v2}) L(v)=({og2},{ov2}) | w=({f1},{n1}) L(w)=({of1},{on1})")
        log_message(f"QR parsed — c1={c1}, c2={c2} | c1*L(u)=({c1og1},{c1ov1}) c2*L(v)=({c2og2},{c2ov2})")
        log_message(f"Expected answers: {config.expected_answers}")

        config.problem_loaded = True
        log_message(f"Problem successfully initialized via {code_type}! Target: {config.target_vector}")

        # Adding a 3-second delay after the QR code has successfully scanned
        time.sleep(3.0)

    except Exception as e:
        log_message(f"CRITICAL QR SCAN FAILURE: {str(e)}")

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
        cv2.imwrite(os.path.join(captures_dir, f"p{config.problem_number}_warped_{config.current_step}_{stamp}.png"), local_sheet)
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
            resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
            
            # Localized grid normalization via CLAHE to erase overhead lighting glare
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            enhanced = clahe.apply(resized)
            
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            try:
                cv2.imwrite(os.path.join(captures_dir, f"p{config.problem_number}_{config.current_step}_{region_name}_{stamp}.png"), binary)
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
        if not all_empty and expected:
            # A region passes if every expected value is present in what was
            # recognized for that region (duplicates counted individually via
            # Counter), even if extra/unexpected tokens were also picked up.
            is_valid = all(
                not (Counter(expected_tokens) - Counter(recognized_data.get(region_name, [])))
                for region_name, expected_tokens in expected.items()
            )
        if is_valid:
            config.green_count += 1
            config.feedback_step = config.current_step
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
            config.feedback_step = config.current_step
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

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tracking_entry = f"[{timestamp}] OCR TRACK [{config.current_step}] regions -> {list(recognized_data.values())} paper_detected={config.paper_detected}\n"
        with open(config.LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(tracking_entry)
    except Exception:
        pass

    config.is_processing = False

def transform_to_projection_space(w_x, w_y, center_rect, track_mat=None):
    track_mat = track_mat if track_mat is not None else config.tracking_matrix
    if track_mat is None:
        return None
    src_point = np.array([[[w_x, w_y]]], dtype="float32")
    transformed = cv2.perspectiveTransform(src_point, track_mat)
    cam_x = transformed[0][0][0]
    cam_y = transformed[0][0][1]
    p_x = center_rect.x + round((cam_x / config.CAM_W) * center_rect.width)
    p_y = center_rect.y + round((cam_y / config.CAM_H) * center_rect.height)
    return p_x, p_y

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
    sx = int(((cx / config.CAM_W - 0.5) * config.pen_accel + 0.5) * screen_w)
    sx = max(0, min(screen_w - 1, sx))
    sy = int(((cy / config.CAM_H - 0.5) * config.pen_accel + 0.5) * screen_h)
    sy = max(0, min(screen_h - 1, sy))
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
    if screen_pos is not None:
        config.pen_track_last = screen_pos
        config.pen_position = screen_pos
        config.pen_visible = True
    else:
        config.pen_track_active = False
        config.pen_position = None
        config.pen_visible = False
        config.pen_smooth_pos = None
        config.pen_median_buffer = []
        config.pen_hover_button = None


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
