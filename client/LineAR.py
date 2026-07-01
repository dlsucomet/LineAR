import sys
import os
import re
import cv2
import numpy as np
import pygame
import threading
import easyocr
import argparse
import json
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["highlights", "no_highlights"], default="no_highlights")
args = parser.parse_args()

pygame.init()
pygame.font.init()

# 1920 x 1080
# 1600 x 900
# 1280 x 720
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
TOP_BAR_HEIGHT = 50
BOTTOM_BAR_HEIGHT = 70
OUTER_GAP = 14
INNER_GAP = 14

COLOR_BG = (248, 250, 252)
COLOR_TEXT = (0, 0, 0)
COLOR_AXIS = (0, 0, 0)
COLOR_BLUE = (26, 58, 107)
COLOR_GRID = (226, 232, 240)
COLOR_WHITE = (255, 255, 255)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("LineAR - Production Projector Space")
clock = pygame.time.Clock()

font_large = pygame.font.SysFont("segoeui", 24, bold=True)
font_medium = pygame.font.SysFont("segoeui", 18)
font_body = pygame.font.SysFont("segoeui", 15)
font_bold = pygame.font.SysFont("segoeui", 15, bold=True)
font_equation = pygame.font.SysFont("segoeui", 22, bold=True)

current_step = "secondStepLeft"

status_msg = "System Ready. Place paper to align ArUco markers."
is_processing = False

active_vectors = [
    {"x": 1, "y": 2, "label": "u"},
    {"x": 0, "y": 1, "label": "w"}
]

warped_document = None
tracking_matrix = None
paper_detected = False
shared_frame_lock = threading.Lock()
fullscreen = False
app_phase = "start"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

existing_folders = sorted(
    (d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d)) and re.match(r'^p\d+$', d)),
    key=lambda x: int(re.search(r'p(\d+)', x).group(1))
)
p_num = (int(re.search(r'p(\d+)', existing_folders[-1]).group(1)) + 1) if existing_folders else 1
PARTICIPANT_DIR = os.path.join(LOGS_DIR, f"p{p_num}")
os.makedirs(PARTICIPANT_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(PARTICIPANT_DIR, "linear_session.log")
SESSION_PATH = os.path.join(PARTICIPANT_DIR, f"p{p_num}.json")
def log_message(message):
    """Updates the internal UI status and safely writes the entry into the local text log file."""
    global status_msg
    status_msg = message  
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}\n"
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed writing to file log: {str(e)}", file=sys.stderr)
with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
    f.write(f"=== LineAR System Session Log Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
with open(SESSION_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mode": getattr(args, 'mode', 'projector'),
        "time_started": datetime.now().strftime("%H:%M:%S"),
        "time_ended": None,
        "end_task_pressed": False,
        "green_count": 0,
        "red_count": 0,
    }, f, indent=2)

def log_message(message):
    global status_msg
    status_msg = message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}\n"
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed writing to file log: {str(e)}", file=sys.stderr)


with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
    f.write(f"=== LineAR System Session Log Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

CROP_REGIONS = {
    "secondStepLeft": {
        "one": {"top": 400, "left": 200, "width": 90, "height": 190},
        "two": {"top": 380, "left": 350, "width": 90, "height": 190}
    }
}


def paper_tracking_daemon():
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


def background_ocr_pipeline():
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
        if recognized_data.get("one") == ["12", "-3"] and recognized_data.get("two") == ["2", "-4"]:
            active_vectors.append({"x": 14, "y": -7, "label": "Final Proj"})
            log_message("SUCCESS: Student calculations verified as mathematically valid.")
            current_step = "complete"
        else:
            log_message(f"REJECTED: Problem submission mismatch. Detected data: {list(recognized_data.values())}")
    except Exception as e:
        log_message(f"CRITICAL OCR FAILURE: Exception thrown -> {str(e)}")
    is_processing = False


def start_session():
    global app_phase, ocr_reader, cap
    log_message("Loading OCR Engine context...")
    ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    log_message("OCR Engine Ready.")
    log_message("Initializing hardware camera capture access...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        log_message("CRITICAL ERROR: Could not open the system video capture stream.")
    log_message("Booting up backend real-time tracking daemon thread pass...")
    threading.Thread(target=paper_tracking_daemon, daemon=True).start()
    app_phase = "running"
    with open(SESSION_PATH) as f:
        data = json.load(f)
    data["time_started"] = datetime.now().strftime("%H:%M:%S")
    with open(SESSION_PATH, "w") as f:
        json.dump(data, f, indent=2)


def handle_shutdown(from_button=False):
    global running
    running = False
    if from_button and os.path.exists(SESSION_PATH):
        with open(SESSION_PATH) as f:
            data = json.load(f)
        data["end_task_pressed"] = True
        with open(SESSION_PATH, "w") as f:
            json.dump(data, f, indent=2)


def transform_to_projection_space(w_x, w_y):
    if tracking_matrix is None:
        return None
    src_point = np.array([[[w_x, w_y]]], dtype="float32")
    transformed = cv2.perspectiveTransform(src_point, tracking_matrix)
    cam_x = transformed[0][0][0]
    cam_y = transformed[0][0][1]
    _, center_rect, _ = get_panel_rects()
    p_x = center_rect.x + int((cam_x / 1280.0) * center_rect.width)
    p_y = center_rect.y + int((cam_y / 720.0) * center_rect.height)
    return int(p_x), int(p_y)


def get_panel_rects():
    W = screen.get_width()
    H = screen.get_height()
    avail_w = W - OUTER_GAP * 2 - INNER_GAP * 2
    panel_y = TOP_BAR_HEIGHT + OUTER_GAP
    panel_h = H - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT - OUTER_GAP * 2
    left_w = int(avail_w * 0.29)
    center_w = int(avail_w * 0.42)
    right_w = avail_w - left_w - center_w
    x = OUTER_GAP
    left = pygame.Rect(x, panel_y, left_w, panel_h)
    x += left_w + INNER_GAP
    center = pygame.Rect(x, panel_y, center_w, panel_h)
    x += center_w + INNER_GAP
    right = pygame.Rect(x, panel_y, right_w, panel_h)
    return left, center, right


def toggle_fullscreen():
    global fullscreen, WINDOW_WIDTH, WINDOW_HEIGHT, screen
    if fullscreen:
        info = pygame.display.Info()
        WINDOW_WIDTH = info.current_w
        WINDOW_HEIGHT = info.current_h
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
    else:
        # 1920 x 1080
        # 1280 x 720
        WINDOW_WIDTH = 1280
        WINDOW_HEIGHT = 720
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        test = current + (" " if current else "") + word
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_top_bar(surface):
    W = surface.get_width()
    bar_rect = pygame.Rect(0, 0, W, TOP_BAR_HEIGHT)
    pygame.draw.rect(surface, COLOR_BLUE, bar_rect)
    text = font_large.render("Place paper on the designated projection area", True, COLOR_WHITE)
    text_rect = text.get_rect(center=(W // 2, TOP_BAR_HEIGHT // 2))
    surface.blit(text, text_rect)
    badge_text = font_body.render(status_msg, True, COLOR_WHITE)
    pad = 10
    badge_w = badge_text.get_width() + pad * 2
    badge_h = 26
    badge_x = W - badge_w - 16
    badge_y = (TOP_BAR_HEIGHT - badge_h) // 2
    pygame.draw.rect(surface, COLOR_TEXT, (badge_x, badge_y, badge_w, badge_h))
    pygame.draw.rect(surface, COLOR_WHITE, (badge_x, badge_y, badge_w, badge_h), 1)
    surface.blit(badge_text, (badge_x + pad, badge_y + (badge_h - badge_text.get_height()) // 2))


def draw_bottom_bar(surface):
    W = surface.get_width()
    H = surface.get_height()
    bar_rect = pygame.Rect(0, H - BOTTOM_BAR_HEIGHT, W, BOTTOM_BAR_HEIGHT)
    pygame.draw.rect(surface, COLOR_BG, bar_rect)
    console = font_bold.render(f"CONSOLE LOG: {status_msg}", True, COLOR_TEXT)
    surface.blit(console, (OUTER_GAP, H - BOTTOM_BAR_HEIGHT + (BOTTOM_BAR_HEIGHT - console.get_height()) // 2))


def draw_cartesian_plane(surface, area):
    pygame.draw.rect(surface, COLOR_WHITE, area)
    ox = area.x + area.width // 2
    oy = area.y + area.height // 2
    scale = max(1, int(min(area.width, area.height) / 14))
    grid_surf = pygame.Surface((area.width, area.height), pygame.SRCALPHA)
    for x in range(ox % scale, area.x + area.width, scale):
        pygame.draw.line(grid_surf, (*COLOR_GRID, 180), (x, area.y), (x, area.y + area.height))
    for y in range(oy % scale, area.y + area.height, scale):
        pygame.draw.line(grid_surf, (*COLOR_GRID, 180), (area.x, y), (area.x + area.width, y))
    surface.blit(grid_surf, (area.x, area.y))
    pygame.draw.line(surface, COLOR_TEXT, (area.x, oy), (area.x + area.width, oy), 2)
    pygame.draw.line(surface, COLOR_TEXT, (ox, area.y), (ox, area.y + area.height), 2)
    clip_rect = surface.get_clip()
    surface.set_clip(area)
    for vec in active_vectors:
        tx = ox + int(vec["x"] * scale)
        ty = oy - int(vec["y"] * scale)
        if tx < area.x or tx > area.x + area.width or ty < area.y or ty > area.y + area.height:
            continue
        pygame.draw.line(surface, COLOR_TEXT, (ox, oy), (tx, ty), 4)
        pygame.draw.circle(surface, COLOR_TEXT, (tx, ty), 5)
        label = f'{vec["label"]} ({vec["x"]},{vec["y"]})'
        txt = font_bold.render(label, True, COLOR_TEXT)
        surface.blit(txt, (tx + 8, ty - 4))
    surface.set_clip(clip_rect)


def draw_dashed_rect(surface, color, rect, width, dash_len, gap_len):
    x, y, w, h = rect
    segments = []
    for sx in range(x, x + w, dash_len + gap_len):
        ex = min(sx + dash_len, x + w)
        segments.append(((sx, y), (ex, y)))
        segments.append(((sx, y + h), (ex, y + h)))
    for sy in range(y, y + h, dash_len + gap_len):
        ey = min(sy + dash_len, y + h)
        segments.append(((x, sy), (x, ey)))
        segments.append(((x + w, sy), (x + w, ey)))
    for start, end in segments:
        pygame.draw.line(surface, color, start, end, width)


def draw_ar_overlay(surface, area, step):
    dash_len = 6
    gap_len = 4
    target = None
    if step == "secondStepLeft":
        target = (30, 180, 80, 100)
    if target:
        rx = area.x + int(target[0] * area.width / 500)
        ry = area.y + int(target[1] * area.height / 500)
        rw = max(2, int(target[2] * area.width / 500))
        rh = max(2, int(target[3] * area.height / 500))
        draw_dashed_rect(surface, COLOR_TEXT, (rx, ry, rw, rh), 3, dash_len, gap_len)


def draw_instruction_panel(surface, area):
    pad = 20
    content_x = area.x + pad
    content_y = area.y + pad
    max_w = area.width - pad * 2
    heading = font_large.render("Vector Addition Property", True, COLOR_TEXT)
    surface.blit(heading, (content_x, content_y))
    content_y += 40
    text_lines = wrap_text(
        "Linear transformations distribute over scalar values and vector groupings natively.",
        font_body, max_w
    )
    for line in text_lines:
        rendered = font_body.render(line, True, COLOR_TEXT)
        surface.blit(rendered, (content_x, content_y))
        content_y += 22
    content_y += 10
    eq_card_rect = pygame.Rect(content_x, content_y, max_w, 50)
    pygame.draw.rect(surface, COLOR_WHITE, eq_card_rect)
    pygame.draw.rect(surface, COLOR_TEXT, eq_card_rect, 2)
    eq = font_equation.render("[ a ] + [ c ]  =  [ a + c ]", True, COLOR_TEXT)
    eq_rect = eq.get_rect(center=eq_card_rect.center)
    surface.blit(eq, eq_rect)


def draw_panels(surface):
    left_rect, center_rect, right_rect = get_panel_rects()
    for rect in [left_rect, center_rect, right_rect]:
        pygame.draw.rect(surface, COLOR_WHITE, rect)
        pygame.draw.rect(surface, COLOR_BLUE, rect, 3)
    draw_cartesian_plane(surface, left_rect)
    pygame.draw.rect(surface, COLOR_BLUE, left_rect, 3)
    paper_margin = 10
    cam_area = pygame.Rect(
        center_rect.x + paper_margin, center_rect.y + paper_margin,
        center_rect.width - paper_margin * 2, center_rect.height - paper_margin * 2
    )
    paper_aspect = 1.41
    paper_w = min(cam_area.width, int(cam_area.height * paper_aspect))
    paper_h = int(paper_w / paper_aspect)
    if paper_h > cam_area.height:
        paper_h = cam_area.height
        paper_w = int(paper_h * paper_aspect)
    paper_area = pygame.Rect(
        cam_area.x + (cam_area.width - paper_w) // 2,
        cam_area.y + (cam_area.height - paper_h) // 2,
        paper_w, paper_h
    )
    if current_step == "secondStepLeft":
        with shared_frame_lock:
            currently_tracking = paper_detected
        if currently_tracking:
            regions = CROP_REGIONS["secondStepLeft"]
            for r_name, box in regions.items():
                top_left = transform_to_projection_space(box["left"], box["top"])
                bottom_right = transform_to_projection_space(box["left"] + box["width"], box["top"] + box["height"])
                if top_left and bottom_right:
                    box_w = bottom_right[0] - top_left[0]
                    box_h = bottom_right[1] - top_left[1]
                    pygame.draw.rect(surface, COLOR_AXIS, (top_left[0], top_left[1], box_w, box_h), 2)
        else:
            msg = font_bold.render("[ Align ArUco Markers to Project Guides ]", True, COLOR_TEXT)
            msg_rect = msg.get_rect(center=center_rect.center)
            surface.blit(msg, msg_rect)
    else:
        draw_ar_overlay(surface, paper_area, current_step)
    draw_instruction_panel(surface, right_rect)


def draw_panels_start(surface):
    left_rect, center_rect, right_rect = get_panel_rects()
    for rect in [left_rect, center_rect, right_rect]:
        pygame.draw.rect(surface, COLOR_WHITE, rect)
        pygame.draw.rect(surface, COLOR_BLUE, rect, 3)


def draw_start_button(surface, center_rect):
    btn_w, btn_h = 260, 70
    btn_rect = pygame.Rect(center_rect.centerx - btn_w // 2,
                           center_rect.centery - btn_h // 2, btn_w, btn_h)
    mx, my = pygame.mouse.get_pos()
    hovered = btn_rect.collidepoint(mx, my)
    if hovered:
        pygame.draw.rect(surface, COLOR_WHITE, btn_rect, border_radius=12)
        pygame.draw.rect(surface, COLOR_BLUE, btn_rect, 3, border_radius=12)
        text = font_large.render("START", True, COLOR_BLUE)
    else:
        pygame.draw.rect(surface, COLOR_BLUE, btn_rect, border_radius=12)
        text = font_large.render("START", True, COLOR_WHITE)
    text_rect = text.get_rect(center=btn_rect.center)
    surface.blit(text, text_rect)
    return btn_rect


def draw_end_task_button(surface, center_rect):
    btn_w, btn_h = 160, 44
    btn_x = center_rect.centerx - btn_w // 2
    btn_y = screen.get_height() - BOTTOM_BAR_HEIGHT + (BOTTOM_BAR_HEIGHT - btn_h) // 2
    btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
    mx, my = pygame.mouse.get_pos()
    hovered = btn_rect.collidepoint(mx, my)
    if hovered:
        pygame.draw.rect(surface, COLOR_WHITE, btn_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BLUE, btn_rect, 3, border_radius=6)
        text = font_medium.render("End Task", True, COLOR_BLUE)
    else:
        pygame.draw.rect(surface, COLOR_BLUE, btn_rect, border_radius=6)
        text = font_medium.render("End Task", True, COLOR_WHITE)
    text_rect = text.get_rect(center=btn_rect.center)
    surface.blit(text, text_rect)
    return btn_rect


running = True

start_btn_rect = None
end_btn_rect = None

while running:
    screen.fill(COLOR_BG)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            handle_shutdown(from_button=False)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if app_phase == "start" and start_btn_rect and start_btn_rect.collidepoint(event.pos):
                start_session()
            elif app_phase == "running" and end_btn_rect and end_btn_rect.collidepoint(event.pos):
                handle_shutdown(from_button=True)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                handle_shutdown(from_button=False)
            elif event.key == pygame.K_RETURN:
                if app_phase == "start":
                    start_session()
                elif not is_processing:
                    is_processing = True
                    threading.Thread(target=background_ocr_pipeline, daemon=True).start()
            elif event.key == pygame.K_F11:
                toggle_fullscreen()

    if app_phase == "start":
        draw_top_bar(screen)
        draw_panels_start(screen)
        _, center_rect, _ = get_panel_rects()
        start_btn_rect = draw_start_button(screen, center_rect)
        draw_bottom_bar(screen)
    else:
        draw_top_bar(screen)
        draw_panels(screen)
        draw_bottom_bar(screen)
        _, center_rect, _ = get_panel_rects()
        end_btn_rect = draw_end_task_button(screen, center_rect)

    pygame.display.flip()
    clock.tick(60)

log_message("System shutting down. Closing capture devices.")
if os.path.exists(SESSION_PATH):
    with open(SESSION_PATH) as f:
        data = json.load(f)
    data["time_ended"] = datetime.now().strftime("%H:%M:%S")
    with open(SESSION_PATH, "w") as f:
        json.dump(data, f, indent=2)
if 'cap' in dir() and cap is not None:
    cap.release()
pygame.quit()
sys.exit()
