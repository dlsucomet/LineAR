import re
import cv2
import numpy as np
import config
from logger import log_message

def paper_tracking_daemon():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    max_w, max_h = 1000, 1200
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
            corner_map = {int(ids[i][0]): corners[i][0] for i in range(len(ids))}
            if all(k in corner_map for k in [0, 1, 2, 3]):
                src_pts = np.array([
                    corner_map[0][0], corner_map[1][0],
                    corner_map[3][0], corner_map[2][0]
                ], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                _, M_inv = cv2.invert(M)
                with config.shared_frame_lock:
                    config.tracking_matrix = M_inv
                    config.warped_document = cv2.warpPerspective(frame, M, (max_w, max_h))
                    config.paper_detected = True
                if not last_state:
                    log_message("Tracking Lock Acquired: Target sheet anchors located.")
                    last_state = True
                continue
        with config.shared_frame_lock:
            config.paper_detected = False
        if last_state:
            log_message("Tracking Lock Lost: Target sheet missing or occluded.")
            last_state = False

def background_ocr_pipeline():
    with config.shared_frame_lock:
        if not config.paper_detected or config.warped_document is None:
            log_message("Verification Failed: Paper target not in sight.")
            config.is_processing = False
            return
        local_sheet = config.warped_document.copy()
        
    log_message(f"Starting OCR evaluation routine for calculation phase: {config.current_step}")
    try:
        regions = config.CROP_REGIONS.get(config.current_step, {})
        recognized_data = {}
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
        
        # Linear algebra step validation
        is_valid = False
        if config.current_step == "firstStepLeft" and recognized_data.get("one") == ["3"]:
            is_valid = True
        elif config.current_step == "firstStepRight" and recognized_data.get("one") == ["-2"]:
            is_valid = True
        elif config.current_step == "secondStepLeft" and recognized_data.get("one") == ["12", "-3"]:
            is_valid = True
        elif config.current_step == "secondStepRight" and recognized_data.get("one") == ["2", "-4"]:
            is_valid = True
        elif config.current_step == "thirdStep" and recognized_data.get("one") == ["14", "-7"]:
            config.active_vectors.append({"x": 14, "y": -7, "label": "L(v)"})
            is_valid = True

        if is_valid:
            idx = config.STEP_SEQUENCE.index(config.current_step)
            config.current_step = config.STEP_SEQUENCE[idx + 1]
            log_message(f"SUCCESS: Moving to step {config.current_step}")
        else:
            log_message(f"REJECTED: Submission mismatch. Got: {list(recognized_data.values())}")
            
    except Exception as e:
        log_message(f"CRITICAL OCR FAILURE: Exception thrown -> {str(e)}")
    config.is_processing = False

def transform_to_projection_space(w_x, w_y, center_rect):
    if config.tracking_matrix is None:
        return None
    src_point = np.array([[[w_x, w_y]]], dtype="float32")
    transformed = cv2.perspectiveTransform(src_point, config.tracking_matrix)
    cam_x = transformed[0][0][0]
    cam_y = transformed[0][0][1]
    p_x = center_rect.x + int((cam_x / 1280.0) * center_rect.width)
    p_y = center_rect.y + int((cam_y / 720.0) * center_rect.height)
    return int(p_x), int(p_y)