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

_OCR_TARGET_WIDTH = 800


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
    smoothed_tracking = None
    ema_alpha = 0.3
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
                if config.markers_visible_since == 0:
                    config.markers_visible_since = time.time()
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
            config.markers_visible_since = 0
            if last_state:
                log_message("Tracking Lock Lost: Target sheet missing or occluded.")
                last_state = False

_DEBUG_CAPTURE_COOLDOWN = 1.0
_last_debug_capture_time = 0.0
_STEP_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0),
]


def debug_crop_capture():
    global _last_debug_capture_time
    now = time.time()
    if now - _last_debug_capture_time < _DEBUG_CAPTURE_COOLDOWN:
        return
    _last_debug_capture_time = now

    with config.shared_frame_lock:
        if config.warped_document is None:
            return
        sheet = config.warped_document.copy()

    captures_dir = os.path.join(config.PARTICIPANT_DIR, "ocr_captures")
    os.makedirs(captures_dir, exist_ok=True)

    annotated = sheet.copy()
    img_h, img_w = annotated.shape[:2]
    use_bands = getattr(config.args, "mode", "") == "no_highlights"

    if use_bands:
        for i, (zone_name, zone) in enumerate(config.STEP_ZONES.items()):
            color = _STEP_COLORS[i % len(_STEP_COLORS)]
            z_top = max(0, min(zone["top"], img_h - 1))
            z_left = max(0, min(zone["left"], img_w - 1))
            z_h = max(1, min(zone["height"], img_h - z_top))
            z_w = max(1, min(zone["width"], img_w - z_left))
            cv2.rectangle(annotated, (z_left, z_top), (z_left + z_w, z_top + z_h), color, 3)
            cv2.putText(annotated, zone_name, (z_left + 4, z_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        zone_key = config.STEP_TO_ZONE.get(config.current_step)
        if zone_key and zone_key in config.STEP_ZONES:
            zone = config.STEP_ZONES[zone_key]
            z_top = max(0, min(zone["top"], img_h - 1))
            z_left = max(0, min(zone["left"], img_w - 1))
            z_h = max(1, min(zone["height"], img_h - z_top))
            z_w = max(1, min(zone["width"], img_w - z_left))
            zone_crop = sheet[z_top:z_top+z_h, z_left:z_left+z_w]
            bands = detect_content_bands(zone_crop)
            for bi, (row_start, row_end, col_start, col_end) in enumerate(bands):
                abs_r0 = z_top + row_start
                abs_r1 = z_top + row_end
                abs_c0 = z_left + col_start
                abs_c1 = z_left + col_end
                cv2.rectangle(annotated, (abs_c0, abs_r0), (abs_c1, abs_r1), (0, 255, 0), 2)
                cv2.putText(annotated, f"band{bi}", (abs_c0 + 2, abs_r0 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(captures_dir,
                        f"debug_crop_{config.current_step}_zone.png"), zone_crop)
    else:
        for i, (step_name, regions) in enumerate(config.CROP_REGIONS.items()):
            color = _STEP_COLORS[i % len(_STEP_COLORS)]
            for region_name, box in regions.items():
                top = max(0, min(box["top"], img_h - 1))
                left = max(0, min(box["left"], img_w - 1))
                r_h = max(1, min(box["height"], img_h - top))
                r_w = max(1, min(box["width"], img_w - left))
                cv2.rectangle(annotated, (left, top), (left + r_w, top + r_h), color, 3)
                label = f"{step_name}:{region_name}"
                cv2.putText(annotated, label, (left + 4, top + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        current_regions = config.CROP_REGIONS.get(config.current_step, {})
        for region_name, box in current_regions.items():
            top = max(0, min(box["top"], img_h - 1))
            left = max(0, min(box["left"], img_w - 1))
            r_h = max(1, min(box["height"], img_h - top))
            r_w = max(1, min(box["width"], img_w - left))
            crop = sheet[top:top + r_h, left:left + r_w]
            cv2.imwrite(os.path.join(captures_dir,
                        f"debug_crop_{config.current_step}_{region_name}.png"), crop)

    cv2.imwrite(os.path.join(captures_dir, "debug_all_regions.png"), annotated)


def detect_content_bands(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_per_row = (binary > 0).sum(axis=1)
    row_threshold = w * 0.01
    min_band_height = 15

    in_content = False
    bands = []
    band_start = 0
    for row in range(h):
        has_content = ink_per_row[row] > row_threshold
        if has_content and not in_content:
            band_start = row
            in_content = True
        elif not has_content and in_content:
            band_end = row
            in_content = False
            if band_end - band_start >= min_band_height:
                band_region = binary[band_start:band_end, :]
                cols_with_ink = np.where(band_region.max(axis=0) > 0)[0]
                if len(cols_with_ink) > 0:
                    bands.append((band_start, band_end, cols_with_ink.min(), cols_with_ink.max()))
    if in_content:
        band_end = h
        if band_end - band_start >= min_band_height:
            band_region = binary[band_start:band_end, :]
            cols_with_ink = np.where(band_region.max(axis=0) > 0)[0]
            if len(cols_with_ink) > 0:
                bands.append((band_start, h, cols_with_ink.min(), cols_with_ink.max()))
    return bands


def _ocr_region(region_img, reader, min_confidence=0.3):
    result = reader.predict(input=region_img)
    tokens = []
    conf_map = {}
    if result and len(result) > 0:
        for text, score in zip(result[0]['rec_texts'], result[0]['rec_scores']):
            if score < min_confidence:
                continue
            cleaned = text.replace(" ", "")
            for match in re.finditer(r'-?\d+', cleaned):
                token = match.group()
                if token and token != "-":
                    tokens.append(token)
                    conf_map[token] = score
    return tokens, conf_map, "paddleocr", region_img


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
            "firstStep":        {},
            "firstStepOne":     {"one": [str(f1), str(n1)]},
            "firstStepTwo":     {"one": [str(g1), str(v1)]},
            "firstStepThree":   {"one": [str(g2), str(v2)]},
            "firstStepFour":    {"one": [str(c1), str(c2)]},
            "secondStepLeft":   {"one": [str(c1), str(og1), str(ov1)]},
            "secondStepRight":  {"one": [str(c2), str(og2), str(ov2)]},
            "thirdStepLeft":    {"one": [str(c1og1), str(c1ov1)]},
            "thirdStepRight":   {"one": [str(c2og2), str(c2ov2)]},
            "fourthStep":       {"one": [str(of1), str(on1)]},
        }

        log_message(f"QR parsed — u=({g1},{v1}) L(u)=({og1},{ov1}) | v=({g2},{v2}) L(v)=({og2},{ov2}) | w=({f1},{n1}) L(w)=({of1},{on1})")
        log_message(f"QR parsed — c1={c1}, c2={c2} | c1*L(u)=({c1og1},{c1ov1}) c2*L(v)=({c2og2},{c2ov2})")
        log_message(f"Expected answers: {config.expected_answers}")

        config.problem_loaded = True
        log_message(f"Problem successfully initialized via {code_type}! Target: {config.target_vector}")

    except Exception as e:
        log_message(f"CRITICAL QR SCAN FAILURE: {str(e)}")

    config.is_processing = False



def _expected_subset_match(recognized_data, expected):
    band_tokens = [str(t) for t in recognized_data.get("bands", [])]
    for region_name, expected_tokens in expected.items():
        recognized_tokens = [str(t) for t in recognized_data.get(region_name, [])]
        if not recognized_tokens:
            recognized_tokens = band_tokens
        for token in expected_tokens:
            if str(token) not in recognized_tokens:
                return False
    return True


def background_ocr_pipeline():
    with config.shared_frame_lock:
        if config.warped_document is None:
            log_message("Verification Failed: No warped document data available yet.")
            config.last_ocr_finish_time = time.time()
            config.blank_projection = False
            config.is_processing = False
            return
        if not config.paper_detected:
            log_message("WARNING: Paper not currently in view — reusing last known warp.")
        local_sheet = config.warped_document.copy()

    captures_dir = os.path.join(config.PARTICIPANT_DIR, "ocr_captures")
    os.makedirs(captures_dir, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S_%f")[:-3]

    log_message(f"Starting OCR evaluation routine for calculation phase: {config.current_step}")
    recognized_data = {}
    use_bands = getattr(config.args, "mode", "") == "no_highlights"
    try:
        img_h, img_w = local_sheet.shape[:2]

        if use_bands and config.current_step in config.STEP_TO_ZONE:
            zone = config.STEP_ZONES[config.STEP_TO_ZONE[config.current_step]]
            z_top = max(0, min(zone["top"], img_h - 1))
            z_left = max(0, min(zone["left"], img_w - 1))
            z_h = max(1, min(zone["height"], img_h - z_top))
            z_w = max(1, min(zone["width"], img_w - z_left))
            zone_crop = local_sheet[z_top:z_top+z_h, z_left:z_left+z_w]

            bands = detect_content_bands(zone_crop)
            log_message(f"Zone '{config.STEP_TO_ZONE[config.current_step]}': found {len(bands)} ink bands")

            all_tokens = []
            all_conf = {}
            for bi, (row_start, row_end, col_start, col_end) in enumerate(bands):
                padding = 10
                r0 = max(0, row_start - padding)
                r1 = min(zone_crop.shape[0], row_end + padding)
                c0 = max(0, col_start - padding)
                c1 = min(zone_crop.shape[1], col_end + padding)
                band_crop = zone_crop[r0:r1, c0:c1]

                scale = _OCR_TARGET_WIDTH / band_crop.shape[1] if band_crop.shape[1] > _OCR_TARGET_WIDTH else 1.0
                if scale < 1.0:
                    band_crop = cv2.resize(band_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                tokens, conf_map, winning, debug_img = _ocr_region(band_crop, config.ocr_reader)
                log_message(f"Band {bi} (rows {row_start}-{row_end}): winning={winning} tokens={tokens}")
                all_tokens.extend(tokens)
                all_conf.update(conf_map)

                try:
                    if debug_img is not None:
                        cv2.imwrite(os.path.join(captures_dir,
                                    f"p{config.problem_number}_{config.current_step}_band{bi}_{stamp}.png"), debug_img)
                except Exception:
                    pass

            recognized_data["bands"] = all_tokens
            log_message(f"All bands tokens={all_tokens} confidences={all_conf}")

        else:
            regions = config.CROP_REGIONS.get(config.current_step, {})
            for region_name, box in regions.items():
                top, left, width, height = box["top"], box["left"], box["width"], box["height"]
                r_top = max(0, min(top, img_h - 1))
                r_left = max(0, min(left, img_w - 1))
                r_height = max(1, min(height, img_h - r_top))
                r_width = max(1, min(width, img_w - r_left))
                crop = local_sheet[r_top:r_top+r_height, r_left:r_left+r_width]

                scale = _OCR_TARGET_WIDTH / crop.shape[1] if crop.shape[1] > _OCR_TARGET_WIDTH else 1.0
                if scale < 1.0:
                    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                tokens, conf_map, winning, debug_img = _ocr_region(crop, config.ocr_reader)
                recognized_data[region_name] = tokens

                try:
                    if debug_img is not None:
                        cv2.imwrite(os.path.join(captures_dir, f"p{config.problem_number}_{config.current_step}_{region_name}_{stamp}.png"), debug_img)
                except Exception:
                    pass

                log_message(f"Region '{region_name}': winning={winning} tokens={tokens} confidences={conf_map}")
        
        is_valid = False
        expected = config.expected_answers.get(config.current_step, {})
        all_empty = all(len(v) == 0 for v in recognized_data.values())

        if not expected and not regions:
            idx = config.STEP_SEQUENCE.index(config.current_step)
            if idx < len(config.STEP_SEQUENCE) - 1:
                config.current_step = config.STEP_SEQUENCE[idx + 1]
                log_message(f"SKIPPED: Step has no expected values, advancing to '{config.current_step}'.")
            config.last_ocr_finish_time = time.time()
            config.blank_projection = False
            config.is_processing = False
            return

        if not all_empty and expected and _expected_subset_match(recognized_data, expected):
            is_valid = True
        if is_valid:
            config.green_count += 1
            with config.feedback_lock:
                config.feedback_step = config.current_step
                config.feedback_state = "green"
                config.feedback_timer = 60
                config.show_hint = False
            idx = config.STEP_SEQUENCE.index(config.current_step)
            if idx < len(config.STEP_SEQUENCE) - 1:
                config.current_step = config.STEP_SEQUENCE[idx + 1]
                log_message(f"SUCCESS: Moving to step {config.current_step} | Expected: {expected} | Got: {recognized_data}")
            else:
                config.app_phase = "nasa_tlx"
                log_message(f"SUCCESS: Problem completed. Starting NASA-TLX questionnaire. | Expected: {expected} | Got: {recognized_data}")
        else:
            config.red_count += 1
            with config.feedback_lock:
                config.feedback_step = config.current_step
                config.feedback_state = "red"
                config.feedback_timer = 60
                config.show_hint = True
            log_message(f"REJECTED: Step '{config.current_step}' incorrect | Expected: {expected} | Got: {recognized_data}")
        
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

    config.last_ocr_finish_time = time.time()
    config.blank_projection = False
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