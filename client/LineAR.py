import sys
import os
import re
import cv2
import numpy as np
import pygame
import threading
import easyocr
from datetime import datetime  # Added for high-precision log timestamps

# Initialize Pygame Core
pygame.init()
pygame.font.init()

# Projector Optimization (High-Contrast Monochrome Layout)
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 800
COLOR_BG = (255, 255, 255)
COLOR_TEXT = (0, 0, 0)
COLOR_AXIS = (0, 0, 0)
COLOR_GRID = (240, 240, 240)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("LineAR - Production Projector Space")
clock = pygame.time.Clock()

# Strict Layout Dimensions
PANEL_Y = 50
PANEL_HEIGHT = 650
LEFT_X, LEFT_WIDTH = 20, 420
CENTER_X, CENTER_WIDTH = 460, 520
RIGHT_X, RIGHT_WIDTH = 1000, 420

# Type Scales
font_title = pygame.font.SysFont("segoeui", 18, bold=True)
font_body = pygame.font.SysFont("segoeui", 14, bold=False)
font_bold = pygame.font.SysFont("segoeui", 14, bold=True)

# System States
current_step = "secondStepLeft"
status_msg = "System Ready. Place paper to align ArUco markers."
is_processing = False

# Active Coordinate States
active_vectors = [
    {"x": 1, "y": 2, "label": "u"},
    {"x": 0, "y": 1, "label": "w"}
]

# Shared Global Coordinates for Real-time Tracking
warped_document = None
tracking_matrix = None  
paper_detected = False
shared_frame_lock = threading.Lock()

# Define the log destination path
LOG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "linear_session.log"))

# ────────────────────────────────────────────────────────
# FILE LOGGING UTILITY SYSTEM
# ────────────────────────────────────────────────────────
def log_message(message):
    """Updates the internal UI status and safely writes the entry into a local text log file."""
    global status_msg
    status_msg = message  # Update the display console string
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}\n"
    
    # Thread-safe write execution to disk file
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed writing to file log: {str(e)}", file=sys.stderr)

# Clear old session logs at launch and write initialization header
with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
    f.write(f"=== LineAR System Session Log Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

log_message("Loading OCR Engine context...")
ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
log_message("OCR Engine Ready.")

# Boot Local Capture Devices
log_message("Initializing hardware camera capture access...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    log_message("CRITICAL ERROR: Could not open the system video capture stream.")

# Exact bounding mappings from your backend template configuration
CROP_REGIONS = {
    "secondStepLeft": {
        "one": {"top": 400, "left": 200, "width": 90, "height": 190},
        "two": {"top": 380, "left": 350, "width": 90, "height": 190}
    }
}

# ────────────────────────────────────────────────────────
# CONTINUOUS REAL-TIME TRACKING THREAD
# ────────────────────────────────────────────────────────
def paper_tracking_daemon():
    """Continuously computes the perspective transformation matrix from the webcam feed."""
    global tracking_matrix, paper_detected, warped_document
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    max_w, max_h = 1000, 1200
    dst_pts = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")

    last_state = False

    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        corners, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None and len(ids) >= 4:
            corner_map = {int(ids[i][0]): corners[i][0] for i in range(len(ids))}
            
            if all(k in corner_map for k in [0, 1, 2, 3]):
                src_pts = np.array([
                    corner_map[0][0],  
                    corner_map[1][0],  
                    corner_map[3][0],  
                    corner_map[2][0]   
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                _, M_inv = cv2.invert(M)
                
                with shared_frame_lock:
                    tracking_matrix = M_inv
                    warped_document = cv2.warpPerspective(frame, M, (max_w, max_h))
                    paper_detected = True
                
                if not last_state:
                    log_message("Tracking Lock Acquired: Target sheet anchors located.")
                    last_state = True
                continue
                
        with shared_frame_lock:
            paper_detected = False
            
        if last_state:
            log_message("Tracking Lock Lost: Target sheet missing or occluded.")
            last_state = False

# ────────────────────────────────────────────────────────
# ASYNCHRONOUS HANDWRITING RECOGNITION THREAD
# ────────────────────────────────────────────────────────
def background_ocr_pipeline():
    """Analyzes handwritten inputs from the warped document snippet when triggered."""
    global is_processing, current_step, active_vectors
    
    with shared_frame_lock:
        if not paper_detected or warped_document is None:
            log_message("Verification Failed: Paper target not in sight.")
            is_processing = False
            return
        local_sheet = warped_document.copy()

    log_message(f"Starting OCR evaluation routine for calculation phase: {current_step}")
    try:
        regions = CROP_REGIONS.get(current_step, {})
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
            
            ocr_results = ocr_reader.readtext(processed_crop, allowlist='0123456789-', paragraph=False)
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

        # Algebraic validation logic evaluation
        if recognized_data.get("one") == ["12", "-3"] and recognized_data.get("two") == ["2", "-4"]:
            active_vectors.append({"x": 14, "y": -7, "label": "Final Proj"})
            log_message("SUCCESS: Student calculations verified as mathematically valid.")
            current_step = "complete"
        else:
            log_message(f"REJECTED: Problem submission mismatch. Detected data: {list(recognized_data.values())}")

    except Exception as e:
        log_message(f"CRITICAL OCR FAILURE: Exception thrown -> {str(e)}")
        
    is_processing = False

# ────────────────────────────────────────────────────────
# COORDINATE TRANSFORMATION UTILITY
# ────────────────────────────────────────────────────────
def transform_to_projection_space(w_x, w_y):
    """Transforms coordinates from the warped A4 sheet layout back to the real-world Pygame UI layout."""
    if tracking_matrix is None:
        return None
    src_point = np.array([[[w_x, w_y]]], dtype="float32")
    transformed = cv2.perspectiveTransform(src_point, tracking_matrix)
    cam_x = transformed[0][0][0]
    cam_y = transformed[0][0][1]
    p_x = CENTER_X + (cam_x / 1280.0) * CENTER_WIDTH
    p_y = PANEL_Y + (cam_y / 720.0) * PANEL_HEIGHT
    return int(p_x), int(p_y)

# ────────────────────────────────────────────────────────
# RENDER SUBSYSTEM INTERFACES
# ────────────────────────────────────────────────────────
def draw_text_wrapped(surface, text, pos, max_width, font, color=COLOR_TEXT):
    words = text.split(' ')
    space_w, space_h = font.size(' ')
    x, y = pos
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        if font.size(test_line)[0] < max_width:
            current_line = test_line
        else:
            surface.blit(font.render(current_line, True, color), (x, y))
            y += space_h + 4
            current_line = word + " "
    if current_line:
        surface.blit(font.render(current_line, True, color), (x, y))

def draw_cartesian_plane(surface):
    origin_x = LEFT_X + (LEFT_WIDTH // 2)
    origin_y = PANEL_Y + (PANEL_HEIGHT // 2)
    scale = 30 

    for x in range(LEFT_X, LEFT_X + LEFT_WIDTH, scale):
        pygame.draw.line(surface, COLOR_GRID, (x, PANEL_Y), (x, PANEL_Y + PANEL_HEIGHT), 1)
    for y in range(PANEL_Y, PANEL_Y + PANEL_HEIGHT, scale):
        pygame.draw.line(surface, COLOR_GRID, (LEFT_X, y), (LEFT_X + LEFT_WIDTH, y), 1)

    pygame.draw.line(surface, COLOR_AXIS, (LEFT_X, origin_y), (LEFT_X + LEFT_WIDTH, origin_y), 2)
    pygame.draw.line(surface, COLOR_AXIS, (origin_x, PANEL_Y), (origin_x, PANEL_Y + PANEL_HEIGHT), 2)

    for vec in active_vectors:
        target_x = origin_x + (vec["x"] * scale)
        target_y = origin_y - (vec["y"] * scale)
        pygame.draw.line(surface, COLOR_AXIS, (origin_x, origin_y), (target_x, target_y), 3)
        pygame.draw.circle(surface, COLOR_AXIS, (int(target_x), int(target_y)), 5)
        surface.blit(font_bold.render(f"{vec['label']} ({vec['x']},{vec['y']})", True, COLOR_TEXT), (target_x + 8, target_y - 8))

# ────────────────────────────────────────────────────────
# MAIN EVENT RUNTIME LOOP
# ────────────────────────────────────────────────────────
running = True
log_message("Booting up backend real-time tracking daemon thread pass...")
threading.Thread(target=paper_tracking_daemon, daemon=True).start()

while running:
    screen.fill(COLOR_BG)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_RETURN and not is_processing:
                is_processing = True
                threading.Thread(target=background_ocr_pipeline, daemon=True).start()

    # Layout Separator Blocks
    pygame.draw.rect(screen, COLOR_AXIS, (LEFT_X, PANEL_Y, LEFT_WIDTH, PANEL_HEIGHT), 2)
    pygame.draw.rect(screen, COLOR_AXIS, (CENTER_X, PANEL_Y, CENTER_WIDTH, PANEL_HEIGHT), 2)
    pygame.draw.rect(screen, COLOR_AXIS, (RIGHT_X, PANEL_Y, RIGHT_WIDTH, PANEL_HEIGHT), 2)

    header = font_title.render("PLACE PAPER ON THE DESIGNATED PROJECTION AREA", True, COLOR_TEXT)
    screen.blit(header, (SCREEN_WIDTH // 2 - header.get_width() // 2, 15))
    screen.blit(font_title.render("PROJECTION ALIGNMENT CANVAS", True, COLOR_TEXT), (CENTER_X + 10, PANEL_Y + 10))

    draw_cartesian_plane(screen)

    # DYNAMIC TRACKED AR OVERLAY RENDERING
    if current_step == "secondStepLeft":
        regions = CROP_REGIONS["secondStepLeft"]
        
        with shared_frame_lock:
            currently_tracking = paper_detected

        if currently_tracking:
            for r_name, box in regions.items():
                top_left = transform_to_projection_space(box["left"], box["top"])
                bottom_right = transform_to_projection_space(box["left"] + box["width"], box["top"] + box["height"])
                
                if top_left and bottom_right:
                    box_w = bottom_right[0] - top_left[0]
                    box_h = bottom_right[1] - top_left[1]
                    pygame.draw.rect(screen, COLOR_AXIS, (top_left[0], top_left[1], box_w, box_h), 2)
        else:
            msg = font_bold.render("[ Align ArUco Markers to Project Guides ]", True, COLOR_TEXT)
            screen.blit(msg, (CENTER_X + (CENTER_WIDTH // 2) - msg.get_width() // 2, PANEL_Y + 300))

    # Render Context Instructions Sheet
    draw_text_wrapped(screen, "Vector Addition Property:", (RIGHT_X + 20, PANEL_Y + 60), RIGHT_WIDTH - 40, font_bold)
    draw_text_wrapped(screen, "Linear transformations distribute over scalar values and vector groupings natively.", (RIGHT_X + 20, PANEL_Y + 90), RIGHT_WIDTH - 40, font_body)
    pygame.draw.rect(screen, COLOR_AXIS, (RIGHT_X + 20, PANEL_Y + 150, RIGHT_WIDTH - 40, 60), 2)
    screen.blit(font_bold.render("[ a ] + [ c ]  =  [ a + c ]", True, COLOR_TEXT), (RIGHT_X + 40, PANEL_Y + 170))

    # Live Status Bar Console Info
    screen.blit(font_bold.render(f"CONSOLE LOG: {status_msg}", True, COLOR_TEXT), (20, SCREEN_HEIGHT - 40))

    pygame.display.flip()
    clock.tick(60)

log_message("System shutting down. Closing capture devices.")
cap.release()
pygame.quit()
sys.exit()