import sys
import os
import cv2
import numpy as np
import json
import re
import easyocr
from flask import Flask, request, jsonify

app = Flask(__name__)

print("Initializing EasyOCR Engine...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("EasyOCR Engine Ready!")

CAPTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../captures"))
os.makedirs(CAPTURES_DIR, exist_ok=True)

CROP_REGIONS = {
    "secondStepLeft": {
        "one": {"top": 400, "left": 200, "width": 90, "height": 190},
        "two": {"top": 380, "left": 350, "width": 90, "height": 190}
    },
    "secondStepRight": {
        "one": {"top": 400, "left": 500, "width": 90, "height": 90},
        "two": {"top": 380, "left": 635, "width": 90, "height": 190}
    },
    "thirdStepLeft": {
        "one": {"top": 600, "left": 300, "width": 90, "height": 190}
    },
    "thirdStepRight": {
        "one": {"top": 600, "left": 450, "width": 90, "height": 190}
    },
    "fourthStep": {
        "one": {"top": 850, "left": 300, "width": 90, "height": 190}
    }
}

def unwarp_sheet_with_aruco(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(image)
    
    if ids is None or len(ids) < 4:
        return None, "paper_not_found"

    corner_map = {int(ids[i][0]): corners[i][0] for i in range(len(ids))}
    if not all(k in corner_map for k in [0, 1, 2, 3]):
        return None, "missing_corners"

    src_pts = np.array([
        corner_map[0][0],  
        corner_map[1][0],  
        corner_map[3][0],  
        corner_map[2][0]   
    ], dtype="float32")

    max_width, max_height = 1000, 1200
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped, "success"

@app.route('/process', methods=['POST'])
def process_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON payload"}), 400
            
        image_path = data.get("image_path")
        warped_out_path = data.get("warped_out_path")
        current_step = data.get("current_step")

        if not all([image_path, warped_out_path, current_step]):
            return jsonify({"status": "error", "message": "Missing arguments"}), 400

        if not os.path.exists(image_path):
            return jsonify({"status": "error", "message": f"Source file missing: {image_path}"}), 400

        raw_frame = cv2.imread(image_path)
        if raw_frame is None:
            return jsonify({"status": "error", "message": "Failed to decode input frame image"}), 500

        if current_step not in CROP_REGIONS:
            return jsonify({"status": "error", "message": f"Unknown step token: {current_step}"}), 400

        warped_sheet, status_flag = unwarp_sheet_with_aruco(raw_frame)
        if warped_sheet is None:
            return jsonify({"status": "paper_not_found", "message": f"Tracking failed: {status_flag}"})

        cv2.imwrite(warped_out_path, warped_sheet)

        recognized_results = {}
        base_filename = os.path.basename(image_path)
        regions = CROP_REGIONS[current_step]

        for region_name, box in regions.items():
            top, left, width, height = box["top"], box["left"], box["width"], box["height"]

            img_h, img_w = warped_sheet.shape[:2]
            r_top = max(0, min(top, img_h - 1))
            r_left = max(0, min(left, img_w - 1))
            r_height = max(1, min(height, img_h - r_top))
            r_width = max(1, min(width, img_w - r_left))

            pad_h, pad_w = int(r_height * 0.15), int(r_width * 0.15)
            y1, y2 = max(0, r_top - pad_h), min(img_h, r_top + r_height + pad_h)
            x1, x2 = max(0, r_left - pad_w), min(img_w, r_left + r_width + pad_w)
            
            crop = warped_sheet[y1:y2, x1:x2]
            blue_channel = crop[:, :, 0]

            thresh = cv2.adaptiveThreshold(
                blue_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)
            
            for i in range(1, num_labels):
                comp_w = stats[i, cv2.CC_STAT_WIDTH]
                comp_h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # To remove brackets
                if comp_h > (r_height * 0.50) or comp_w > (r_width * 0.75):
                    cleaned_mask[labels == i] = 0

            # Convert white-on-black mask back into black-on-white text image for EasyOCR
            processed_crop = cv2.bitwise_not(cleaned_mask)

            h, w = processed_crop.shape[:2]
            target_size = 300
            if h > w:
                new_h, new_w = target_size, int(w * (target_size / h))
            else:
                new_w, new_h = target_size, int(h * (target_size / w))
                
            crop_resized = cv2.resize(processed_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            final_canvas = np.full((target_size, target_size), 255, dtype=np.uint8)
            offset_y, offset_x = (target_size - new_h) // 2, (target_size - new_w) // 2
            final_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = crop_resized

            crop_debug_path = os.path.join(CAPTURES_DIR, f"{current_step}-{region_name}-{base_filename}")
            cv2.imwrite(crop_debug_path, final_canvas)

            ocr_results = reader.readtext(final_canvas, allowlist='0123456789-', paragraph=False)
            detected_tokens = []

            for (bbox, text, confidence) in ocr_results:
                cleaned = text.replace(" ", "")
                if not cleaned:
                    continue

                y_top = bbox[0][1]
                y_bottom = bbox[2][1]
                box_height = y_bottom - y_top

                # Regex splitter
                for match in re.finditer(r'-?\d+', cleaned):
                    token = match.group()
                    if token == "" or token == "-":
                        continue
                    
                    start_idx = match.start()
                    end_idx = match.end()
                    str_len = len(cleaned)

                    relative_pos = (start_idx + end_idx) / 2.0
                    token_y_center = y_top + (relative_pos / str_len) * box_height

                    detected_tokens.append((token_y_center, token))

            # Sort detected elements strictly top-to-bottom
            detected_tokens.sort(key=lambda item: item[0])
            recognized_results[region_name] = [token for (_, token) in detected_tokens]

        return jsonify({
            "status": "success",
            "recognized": recognized_results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Starting LineAR Flask Core Microservice on port 5000...")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=False)