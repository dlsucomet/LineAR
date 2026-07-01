import sys
import os
import json
import cv2
import pygame
import threading
import easyocr
from datetime import datetime

import config
import ui
from logger import log_message, init_session_files
from pipeline import paper_tracking_daemon, background_ocr_pipeline

STEP_ORDER = ["firstStepLeft", "firstStepRight", "secondStepLeft", "secondStepRight", "thirdStep", "complete"]

def advance_debug_state():
    try:
        current_idx = STEP_ORDER.index(config.current_step)
        next_idx = (current_idx + 1) % len(STEP_ORDER)
        config.current_step = STEP_ORDER[next_idx]
        log_message(f"Debug forced phase transition to: {config.current_step}")
    except ValueError:
        config.current_step = STEP_ORDER[0]

def start_session():
    log_message("Loading OCR Engine context...")
    config.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    log_message("OCR Engine Ready.")
    log_message("Initializing hardware camera capture access...")
    config.cap = cv2.VideoCapture(0)
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

def toggle_fullscreen():
    if config.fullscreen:
        info = pygame.display.Info()
        pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN | pygame.SCALED)
    else:
        pygame.display.set_mode((1280, 720), pygame.RESIZABLE)

screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("LineAR - Production Projector Space")
clock = pygame.time.Clock()
init_session_files()

# Keep persistent bounding box rects across frames so the event handler can read them
debug_btn_rect = pygame.Rect(0, 0, 0, 0) 
start_btn_rect = pygame.Rect(0, 0, 0, 0)
end_btn_rect = pygame.Rect(0, 0, 0, 0)

while config.running:
    screen.fill(config.COLOR_BG)
    
    # Pre-calculate UI geometry maps BEFORE running event checks
    left_rect, center_rect, right_rect = config.get_panel_rects(screen.get_width(), screen.get_height())
    
    # Dynamically determine where the button rect boundaries are located on this frame
    btn_w, btn_h = 260, 70
    start_btn_rect = pygame.Rect(
        center_rect.x + (center_rect.width - btn_w) // 2,
        center_rect.y + (center_rect.height - btn_h) // 2, 
        btn_w, 
        btn_h
    )
    
    end_box_w, end_box_h = 160, 44
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
            elif config.app_phase == "running" and end_btn_rect.collidepoint(event.pos):
                handle_shutdown(from_button=True)
            elif getattr(config, "debug_mode", False) and debug_btn_rect.collidepoint(event.pos):
                advance_debug_state()
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                config.debug_mode = not config.debug_mode
                log_message(f"Debug interface display set to: {config.debug_mode}")
            elif event.key == pygame.K_ESCAPE:
                handle_shutdown(from_button=False)
            elif event.key == pygame.K_RETURN:
                if config.app_phase == "start":
                    start_session()
                elif not config.is_processing:
                    config.is_processing = True
                    threading.Thread(target=background_ocr_pipeline, daemon=True).start()
            elif event.key == pygame.K_F11:
                config.fullscreen = not config.fullscreen
                toggle_fullscreen()

    ui.draw_top_bar(screen)
    if config.app_phase == "start":
        center_rect = ui.draw_panels(screen, mode="start")
        start_btn_rect = ui.draw_start_button(screen, center_rect)
    else:
        center_rect = ui.draw_panels(screen, mode="running")
        end_btn_rect = ui.draw_end_task_button(screen, center_rect)
        
    returned_debug_rect = ui.draw_bottom_bar(screen)
    if returned_debug_rect:
        debug_btn_rect = returned_debug_rect

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
sys.exit()