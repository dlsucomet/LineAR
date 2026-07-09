import sys
import os
import json
import warnings
import threading
from datetime import datetime

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", message="pkg_resources")

import cv2
import pygame
import easyocr

import config
import ui
from logger import log_message, init_session_files
from pipeline import paper_tracking_daemon, background_ocr_pipeline, ocr_problem_data

STEP_ORDER = ["firstStepLeft", "firstStepRight", "secondStepLeft", "secondStepRight", "thirdStep", "complete"]

def advance_debug_state():
    try:
        current_idx = STEP_ORDER.index(config.current_step)
        next_idx = (current_idx + 1) % len(STEP_ORDER)
        config.current_step = STEP_ORDER[next_idx]
        log_message(f"Debug forced phase transition to: {config.current_step}")
    except ValueError:
        config.current_step = STEP_ORDER[0]

def _write_session_counts():
    with open(config.SESSION_PATH) as f:
        data = json.load(f)
    data["green_count"] = config.green_count
    data["red_count"] = config.red_count
    with open(config.SESSION_PATH, "w") as f:
        json.dump(data, f, indent=2)

def debug_simulate_correct():
    config.green_count += 1
    config.feedback_state = "green"
    config.feedback_timer = 60
    idx = config.STEP_SEQUENCE.index(config.current_step)
    if idx < len(config.STEP_SEQUENCE) - 1:
        config.current_step = config.STEP_SEQUENCE[idx + 1]
        log_message(f"DEBUG: Simulated correct -> step {config.current_step}")
    else:
        config.app_phase = "done"
        log_message("DEBUG: Simulated correct -> problem completed, entering done phase")
    _write_session_counts()

def debug_simulate_incorrect():
    config.red_count += 1
    config.feedback_state = "red"
    config.feedback_timer = 60
    log_message("DEBUG: Simulated incorrect")
    _write_session_counts()

def start_session():
    log_message("Loading OCR Engine context...")
    config.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    log_message("OCR Engine Ready.")
    log_message("Initializing hardware camera capture access...")
    # config.cap = cv2.VideoCapture(0)  # Index 0 doesn't work; built-in webcam is index 1
    config.cap = cv2.VideoCapture(2)  # External REDRAGON camera
    config.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    config.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not config.cap.isOpened():
        log_message("CRITICAL ERROR: Could not open the system video capture stream.")
        
    log_message("Booting up backend real-time tracking daemon thread pass...")
    threading.Thread(target=paper_tracking_daemon, daemon=True).start()
    config.app_phase = "running"
    
    with open(config.SESSION_PATH) as f:
        data = json.load(f)
    data["time_started"] = datetime.now().strftime("%H:%M:%S")
    with open(config.SESSION_PATH, "w") as f:
        json.dump(data, f, indent=2)

def handle_shutdown(from_button=False):
    config.running = False
    if from_button and os.path.exists(config.SESSION_PATH):
        with open(config.SESSION_PATH) as f:
            data = json.load(f)
        data["end_task_pressed"] = True
        with open(config.SESSION_PATH, "w") as f:
            json.dump(data, f, indent=2)

def return_to_launcher():
    log_message("Returning to launcher...")
    handle_shutdown(from_button=False)

def toggle_fullscreen():
    if config.fullscreen:
        info = pygame.display.Info()
        pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN | pygame.SCALED)
    else:
        pygame.display.set_mode((1280, 720), pygame.RESIZABLE)

def main(mode):
    config.init_config()
    config.args.mode = mode
    config.running = True
    config.app_phase = "start"
    config.current_step = "firstStepLeft"
    config.transformation_progress = 0.0
    config.green_count = 0
    config.red_count = 0
    config.feedback_state = None
    config.feedback_timer = 0
    config.is_processing = False
    config.debug_preview = False
    config.status_msg = "System Ready. Place paper to align ArUco markers."
    config.cap = None
    config.ocr_reader = None

    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("LineAR - Production Projector Space")
    clock = pygame.time.Clock()
    init_session_files()
    ui._init_fonts()

    # Keep persistent bounding box rects across frames so the event handler can read them
    debug_btn_rect = pygame.Rect(0, 0, 0, 0) 
    start_btn_rect = pygame.Rect(0, 0, 0, 0)
    end_btn_rect = pygame.Rect(0, 0, 0, 0)
    done_btn_rect = pygame.Rect(0, 0, 0, 0)

    while config.running:
        screen.fill(config.COLOR_BG)
        
        # Pre-calculate UI geometry maps BEFORE running event checks
        left_rect, center_rect, right_rect = config.get_panel_rects(screen.get_width(), screen.get_height())
        
        # Dynamically determine where the button rect boundaries are located on this frame
        start_btn_w = max(100, min(260, int(center_rect.width * 0.30)))
        start_btn_h = max(28, min(70, int(start_btn_w * 70 / 260)))
        start_btn_rect = pygame.Rect(
            center_rect.x + (center_rect.width - start_btn_w) // 2,
            center_rect.y + (center_rect.height - start_btn_h) // 2, 
            start_btn_w, 
            start_btn_h
        )
        
        end_box_w = max(80, min(160, int(center_rect.width * 0.18)))
        end_box_h = max(22, min(44, int(end_box_w * 44 / 160)))
        end_btn_rect = pygame.Rect(
            center_rect.centerx - end_box_w // 2, 
            screen.get_height() - config.BOTTOM_BAR_HEIGHT + (config.BOTTOM_BAR_HEIGHT - end_box_h) // 2, 
            end_box_w, 
            end_box_h
        )
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                handle_shutdown(from_button=False)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if config.app_phase == "start" and start_btn_rect.collidepoint(event.pos):
                    start_session()
                elif config.app_phase == "done" and done_btn_rect.collidepoint(event.pos):
                    return_to_launcher()
                elif config.app_phase == "running" and end_btn_rect.collidepoint(event.pos):
                    handle_shutdown(from_button=True)
                elif getattr(config, "debug_mode", False) and debug_btn_rect.collidepoint(event.pos):
                    advance_debug_state()
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    config.debug_mode = not config.debug_mode
                    log_message(f"Debug interface display set to: {config.debug_mode}")
                elif event.key == pygame.K_h:
                    config.debug_hints = not config.debug_hints
                    log_message(f"Debug hints display set to: {config.debug_hints}")
                elif event.key == pygame.K_g and config.debug_mode and config.app_phase in ("running", "done"):
                    debug_simulate_correct()
                elif event.key == pygame.K_r and config.debug_mode and config.app_phase == "running":
                    debug_simulate_incorrect()
                elif event.key == pygame.K_p and config.debug_mode and config.args.mode == "highlights":
                    config.debug_preview = not config.debug_preview
                    if config.debug_preview:
                        import numpy as np
                        config.tracking_matrix = np.eye(3, dtype="float32")
                        config.paper_detected = True
                        log_message("DEBUG: Highlights preview ON")
                    else:
                        config.tracking_matrix = None
                        config.paper_detected = False
                        log_message("DEBUG: Highlights preview OFF")
                elif event.key == pygame.K_ESCAPE:
                    handle_shutdown(from_button=False)
                elif event.key == pygame.K_RETURN:
                    if config.app_phase == "start":
                        start_session()
                    elif config.app_phase == "done":
                        return_to_launcher()
                    elif not config.is_processing:
                        config.is_processing = True
                        threading.Thread(target=background_ocr_pipeline, daemon=True).start()
                elif event.key == pygame.K_F11:
                    config.fullscreen = not config.fullscreen
                    toggle_fullscreen()

            elif event.type == pygame.VIDEORESIZE:
                if not config.fullscreen:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        if config.app_phase == "running" and not config.is_processing:
            now = pygame.time.get_ticks()
            if now - config.last_ocr_time >= 5000:
                config.last_ocr_time = now
                config.is_processing = True
                threading.Thread(target=background_ocr_pipeline, daemon=True).start()

        if config.app_phase == "running" and not config.problem_loaded and config.paper_detected and config.warped_document is not None and not config.is_processing:
            config.is_processing = True
            threading.Thread(target=ocr_problem_data, daemon=True).start()

        if config.current_step == "complete":
            if config.transformation_progress < 1.0:
                config.transformation_progress += 0.05
        else:
            if config.transformation_progress > 0.0:
                config.transformation_progress -= 0.05

        config.transformation_progress = max(0.0, min(1.0, config.transformation_progress))

        ui.draw_top_bar(screen)
        returned_debug_rect = ui.draw_bottom_bar(screen, center_rect)
        if returned_debug_rect:
            debug_btn_rect = returned_debug_rect

        if config.app_phase == "start":
            center_rect = ui.draw_panels(screen, mode="start")
            start_btn_rect = ui.draw_start_button(screen, center_rect)
        elif config.app_phase == "done":
            center_rect = ui.draw_panels(screen, mode="done")
            done_btn_rect = ui.draw_done_button(screen, center_rect)
        else:
            center_rect = ui.draw_panels(screen, mode="running")
            end_btn_rect = ui.draw_end_task_button(screen, center_rect)

        if config.feedback_timer > 0:
            config.feedback_timer -= 1
            if config.feedback_timer == 0:
                config.feedback_state = None

        pygame.display.flip()
        clock.tick(60)

    log_message("System shutting down. Closing capture devices.")
    if os.path.exists(config.SESSION_PATH):
        with open(config.SESSION_PATH) as f:
            data = json.load(f)
        data["time_ended"] = datetime.now().strftime("%H:%M:%S")
        with open(config.SESSION_PATH, "w") as f:
            json.dump(data, f, indent=2)

    if config.cap is not None:
        config.cap.release()
    pygame.quit()


if __name__ == "__main__":
    main(config.args.mode)