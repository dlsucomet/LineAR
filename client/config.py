import os
import re
import cv2
import json
import pygame
import argparse
import threading
from datetime import datetime
from matrix_utils import VariableMatrix

matrix_engine = VariableMatrix()

# Tracking variable for shape morphing progress (0.0 = Base Shape, 1.0 = Fully Transformed)
transformation_progress = 0.0

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["highlights", "no_highlights"], default="no_highlights")
args = parser.parse_args()

app_phase = "start"
fullscreen = False
debug_mode = False
debug_hints = True
debug_preview = False

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

PARTICIPANT_DIR = None
p_num = None
LOG_FILE_PATH = None
SESSION_PATH = None

def init_config():
    global PARTICIPANT_DIR, p_num, LOG_FILE_PATH, SESSION_PATH
    os.makedirs(LOGS_DIR, exist_ok=True)

    existing_folders = sorted(
        (d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d)) and re.match(r'^p\d+$', d)),
        key=lambda x: int(re.search(r'p(\d+)', x).group(1))
    )
    p_num = (int(re.search(r'p(\d+)', existing_folders[-1]).group(1)) + 1) if existing_folders else 1
    PARTICIPANT_DIR = os.path.join(LOGS_DIR, f"p{p_num}")
    os.makedirs(PARTICIPANT_DIR, exist_ok=True)

    LOG_FILE_PATH = os.path.join(PARTICIPANT_DIR, f"p{p_num}_linear_session.log")
    SESSION_PATH = os.path.join(PARTICIPANT_DIR, f"p{p_num}.json")

    with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(f"=== LineAR System Session Log Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "mode": getattr(args, 'mode', 'no_highlights'),
            "time_started": None,
            "time_ended": None,
            "end_task_pressed": False,
            "green_count": 0,
            "red_count": 0,
            "nasa_tlx": None,
            "ueq_s": None,
            "questionnaires_completed": False,
        }, f, indent=2)

STEP_SEQUENCE = ["firstStepLeft", "firstStepRight", "secondStepLeft", "secondStepRight", "thirdStep", "complete"]
current_step = "firstStepLeft"
last_ocr_time = 0
status_msg = "System Ready. Place paper to align ArUco markers."
is_processing = False
running = True
green_count = 0
red_count = 0
feedback_state = None
feedback_timer = 0

active_vectors = []

warped_document = None
tracking_matrix = None
paper_detected = False

proj_calib_matrix = None
proj_calibrated = False

CALIB_MARKER_SCREEN_POSITIONS = {
    4: (0, 0),
    5: (WINDOW_WIDTH - 1, 0),
    6: (WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1),
    7: (0, WINDOW_HEIGHT - 1),
}
shared_frame_lock = threading.Lock()
cap = None
ocr_reader = None

latest_frame = None
pen_position = None
pen_visible = False
pen_click_queue = []
pen_hsv_lower = (90, 80, 80)
pen_hsv_upper = (130, 255, 255)
pen_track_active = False
pen_track_start = None
pen_track_last = None
pen_smooth_pos = None
pen_median_buffer = []
pen_last_click_time = 0
pen_has_clicked = False
pen_debug_hsv = False
pen_debug_collect = 0
pen_debug_samples = []
pen_calibrating = False
pen_calib_samples = []

PROJECTION_REGIONS = {
    "firstStepLeft": {"one": {"left": 150, "top": 200, "width": 120, "height": 60}},
    "firstStepRight": {"one": {"left": 450, "top": 200, "width": 120, "height": 60}},
    "secondStepLeft": {
        "one": {"left": 200, "top": 400, "width": 110, "height": 210},
        "two": {"left": 350, "top": 380, "width": 110, "height": 210}
    },
    "secondStepRight": {
        "one": {"left": 500, "top": 400, "width": 110, "height": 110},
        "two": {"left": 635, "top": 380, "width": 110, "height": 210}
    },
    "thirdStep": {"one": {"left": 300, "top": 600, "width": 250, "height": 210}}
}

problem_loaded = False
expected_answers = {}
target_vector = None

nasa_tlx_responses = [None] * 6
ueq_s_responses = [None] * 8
nasa_tlx_current_page = 0

_questionnaire_nasa_next_btn = None
_questionnaire_nasa_slider_rect = None

_questionnaire_ueq_submit_btn = None
_questionnaire_ueq_circles = None
_questionnaire_ueq_row_h = 0
_questionnaire_ueq_title_bottom = 0

# "one": {"top": 940, "left": 150, "width": 1000, "height": 200}
CROP_REGIONS = {
    "firstStepLeft": {"one": {"top": 1000, "left": 250, "width": 1800, "height": 440}},
    "firstStepRight": {"one": {"top": 720, "left": 150, "width": 1000, "height": 200}},
    "secondStepLeft": {
        "one": {"top": 800, "left": 400, "width": 180, "height": 380},
        "two": {"top": 760, "left": 700, "width": 180, "height": 380}
    },
    "secondStepRight": {
        "one": {"top": 800, "left": 1000, "width": 180, "height": 180},
        "two": {"top": 760, "left": 1270, "width": 180, "height": 380}
    },
    "thirdStep": {"one": {"top": 1200, "left": 600, "width": 180, "height": 380}}
}

def get_panel_rects(surface_width, surface_height):
    avail_w = surface_width - OUTER_GAP * 2 - INNER_GAP * 2
    panel_y = TOP_BAR_HEIGHT + OUTER_GAP
    panel_h = surface_height - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT - OUTER_GAP * 2
    left_w = int(avail_w * 0.29)
    center_w = int(avail_w * 0.42)
    right_w = avail_w - left_w - center_w
    return (
        pygame.Rect(OUTER_GAP, panel_y, left_w, panel_h),
        pygame.Rect(OUTER_GAP + left_w + INNER_GAP, panel_y, center_w, panel_h),
        pygame.Rect(OUTER_GAP + left_w + INNER_GAP + center_w + INNER_GAP, panel_y, right_w, panel_h)
    )