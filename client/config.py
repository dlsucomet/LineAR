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
CAM_W = 1920
CAM_H = 1080
TOP_BAR_HEIGHT = 50
BOTTOM_BAR_HEIGHT = 70
OUTER_GAP = 14
INNER_GAP = 14

COLOR_BG = (248, 250, 252)
COLOR_TEXT = (0, 0, 0)
COLOR_AXIS = (0, 0, 0)
COLOR_BLUE = (26, 58, 107)
COLOR_COPY_BLUE = (60, 120, 200)
COLOR_HIGHLIGHT_BLUE = (100, 170, 240)
COLOR_GRID = (226, 232, 240)
COLOR_WHITE = (255, 255, 255)
COLOR_POINT_LIGHT = (100, 160, 230)
COLOR_POINT_DARK = (26, 58, 107)

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
            "problems": [],
        }, f, indent=2)

STEP_SEQUENCE = ["firstStep", "firstStepOne", "firstStepTwo", "firstStepThree", "firstStepFour", "secondStepLeft", "secondStepRight", "thirdStepLeft", "thirdStepRight", "fourthStep", "complete"]
current_step = "firstStep"
last_ocr_time = 0
status_msg = "System Ready. Place paper to align ArUco markers."
is_processing = False
running = True
green_count = 0
red_count = 0
feedback_state = None
feedback_timer = 0
feedback_step = None
problem_number = 1
problem_ended_early = False

active_vectors = []
qr_data = None

warped_document = None
tracking_matrix = None
frozen_tracking_matrix = None
paper_detected = False
paper_stable_since = 0

shared_frame_lock = threading.Lock()
cap = None
ocr_reader = None

latest_frame = None
pen_position = None
pen_visible = False
pen_click_queue = []
pen_hsv_lower = (98, 61, 87)
pen_hsv_upper = (125, 142, 153)
pen_accel = 2.0
pen_track_active = False
pen_track_start = None
pen_track_last = None
pen_smooth_pos = None
pen_median_buffer = []
pen_last_click_time = 0
pen_has_clicked = False
pen_hover_button = None
pen_hover_start = 0
pen_debug_hsv = False
pen_debug_collect = 0
pen_debug_samples = []
pen_calibrating = False
pen_calib_samples = []


STEP_GUIDANCE = {
    "firstStep": {
        "title": "Linear Combination Setup",
        "desc": "Express the input vector as a linear combination of the given basis vectors.",
        "math": "L(c1*v1 + c2*v2) = c1*L(v1) + c2*L(v2)"
    },
    "firstStepOne": {
        "title": "Identify the Target",
        "desc": "The vector you need to find is highlighted below.",
        "math": "L(v) = ?"
    },
    "firstStepTwo": {
        "title": "First Basis Vector",
        "desc": "Review the first basis vector and its transformation.",
        "math": "L(u) = ?"
    },
    "firstStepThree": {
        "title": "Second Basis Vector",
        "desc": "Review the second basis vector and its transformation.",
        "math": "L(w) = ?"
    },
    "firstStepFour": {
        "title": "Solve for the coefficients",
        "desc": "Solve for the coefficients",
        "math": "L(w) = ?"
    },
    "secondStepLeft": {
        "title": "Transformation Property Expansion",
        "desc": "Substitute the known transformed vector definitions into your linear combination equation.",
        "math": "c1 * [x1 / y1] + c2 * [x2 / y2]"
    },
    "secondStepRight": {
        "title": "Transformation Property Expansion",
        "desc": "Substitute the known transformed vector definitions into your linear combination equation.",
        "math": "c1 * [x1 / y1] + c2 * [x2 / y2]"
    },
    "thirdStepLeft": {
        "title": "Scalar Multiplication",
        "desc": "Distribute the first scalar coefficient into the left vector component elements.",
        "math": "a * [b / c] = [a * b / a * c]"
    },
    "thirdStepRight": {
        "title": "Scalar Multiplication",
        "desc": "Distribute the second scalar coefficient into the right vector component elements.",
        "math": "a * [b / c] = [a * b / a * c]"
    },
    "fourthStep": {
        "title": "Vector Addition",
        "desc": "Perform row-by-row matrix addition on your scaled vector elements to solve.",
        "math": "[a / b] + [c / d] = [a + c / b + d]"
    },
    "complete": {
        "title": "Problem Completed!",
        "desc": "The linear transformation mapping operations match the coordinate target state vector space outputs.",
        "math": "L(v) = [14 / -7]"
    }
}

PROJECTION_REGIONS = {
    "firstStep": {},
    "firstStepOne": {
        "one": {"left": 100, "top": 200, "width": 300, "height": 200},
    },
    "firstStepTwo": {
        "one": {"left": 600, "top": 200, "width": 300, "height": 200},
    },
    "firstStepThree": {
        "one": {"left": 1100, "top": 200, "width": 300, "height": 200},
    },
    "firstStepFour": {
        "one": {"left": 1800, "top": 750, "width": 250, "height": 150},
        "two": {"left": 2400, "top": 750, "width": 250, "height": 150},
    },
    "secondStepLeft": {"one": {"left": 1800, "top": 750, "width": 250, "height": 150},
                      "two": {"left": 600, "top": 1300, "width": 250, "height": 150},
                      "three": {"left": 900, "top": 1200, "width": 300, "height": 300},
                      },
    "secondStepRight": {"one": {"left": 2400, "top": 750, "width": 250, "height": 150},
                      "two": {"left": 1500, "top": 1300, "width": 250, "height": 150},
                      "three": {"left": 1800, "top": 1200, "width": 300, "height": 300},
                      },
    "thirdStepLeft": {
        "one": {"left": 600, "top": 1200, "width": 700, "height": 300},
        "two": {"left": 300, "top": 1800, "width": 300, "height": 300}
    },
    "thirdStepRight": {
        "one": {"left": 1500, "top": 1200, "width": 700, "height": 300},
        "two": {"left": 800, "top": 1800, "width": 300, "height": 300}
    },
    "fourthStep": {
        "one": {"left": 300, "top": 1700, "width": 2000, "height": 300},
        "two": {"left": 300, "top": 2350, "width": 300, "height": 300}
    }
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
    "firstStep": {},
    "firstStepOne": {"one": {"top": 270, "left": 220, "width": 1700, "height": 750}},
    "firstStepTwo": {"one": {"top": 270, "left": 220, "width": 1700, "height": 750}},
    "firstStepThree": {"one": {"top": 270, "left": 220, "width": 1700, "height": 750}},
    "firstStepFour": {"one": {"top": 270, "left": 220, "width": 1700, "height": 750}},
    "secondStepLeft": {"one": {"top": 1030, "left": 220, "width": 1700, "height": 400}},
    "secondStepRight": {"one": {"top": 1030, "left": 220, "width": 1700, "height": 400}},
    "thirdStepLeft": {"one": {"top": 1530, "left": 220, "width": 1700, "height": 400}},
    "thirdStepRight": {"one": {"top": 1530, "left": 220, "width": 1700, "height": 400}},
    "fourthStep": {"one": {"top": 1930, "left": 220, "width": 1700, "height": 400}},
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