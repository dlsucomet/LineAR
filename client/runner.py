from questionnaires import (
    draw_nasa_tlx,
    handle_nasa_tlx_click,
    draw_ueq_s,
    handle_ueq_s_click,
)
from pipeline import (
    paper_tracking_daemon,
    background_ocr_pipeline,
    scan_qr_from_camera,
)
from logger import log_message, init_session_files
import ui
import config
import easyocr
import pygame
import cv2
import sys
import os
import time
import fcntl
import struct
import json
import warnings
import threading
from datetime import datetime

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", message="pkg_resources")


def save_questionnaire_responses():
    with open(config.SESSION_PATH) as f:
        data = json.load(f)
    data["nasa_tlx"] = config.nasa_tlx_responses
    data["ueq_s"] = config.ueq_s_responses
    data["questionnaires_completed"] = True
    with open(config.SESSION_PATH, "w") as f:
        json.dump(data, indent=2, fp=f)
    log_message("Questionnaire responses saved.")


def save_problem_results():
    now = datetime.now().strftime("%H:%M:%S")
    duration = 0
    if hasattr(config, "_problem_start_time") and config._problem_start_time:
        duration = int(time.time() - config._problem_start_time)
    problem = {
        "problem_number": config.problem_number,
        "green_count": config.green_count,
        "red_count": config.red_count,
        "time_started": getattr(config, "_problem_start_str", None),
        "time_ended": now,
        "duration_seconds": duration,
        "end_task_pressed": config.problem_ended_early,
    }
    with open(config.SESSION_PATH) as f:
        data = json.load(f)
    data.setdefault("problems", []).append(problem)
    with open(config.SESSION_PATH, "w") as f:
        json.dump(data, indent=2, fp=f)
    log_message(
        f"Problem {config.problem_number} results saved (green={config.green_count}, red={
            config.red_count}, duration={duration}s, ended_early={config.problem_ended_early})"
    )


def reset_for_new_problem():
    config.problem_number += 1
    config.current_step = "firstStep"
    config.problem_loaded = False
    config.expected_answers = {}
    config.target_vector = None
    config.active_vectors = []
    config.feedback_state = None
    config.feedback_timer = 0
    config.feedback_step = None
    config.show_hint = False
    config.green_count = 0
    config.red_count = 0
    config.is_processing = False
    config.transformation_progress = 0.0
    config.complete_point_progress = 0.0
    config.last_ocr_time = 0
    config.markers_visible_since = 0
    config.problem_ended_early = False
    config.app_phase = "start"
    log_message(f"=== Problem {config.problem_number} Ready ===")


STEP_ORDER = config.STEP_SEQUENCE


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
    config.show_hint = False
    if config.app_phase in ("running", "done"):
        idx = config.STEP_SEQUENCE.index(config.current_step)
        if idx < len(config.STEP_SEQUENCE) - 1:
            config.current_step = config.STEP_SEQUENCE[idx + 1]
            log_message(
                f"DEBUG: Simulated correct -> step {config.current_step}")
        else:
            config.app_phase = "done"
            log_message(
                "DEBUG: Simulated correct -> problem completed, entering done phase"
            )
    else:
        log_message(f"DEBUG: Correct recorded (green={config.green_count})")
    _write_session_counts()


def debug_simulate_incorrect():
    config.red_count += 1
    config.feedback_state = "red"
    config.feedback_timer = 60
    config.show_hint = True
    log_message(f"DEBUG: Incorrect recorded (red={config.red_count})")
    _write_session_counts()


def start_session():
    if config.ocr_reader is None:
        log_message("Loading OCR Engine context...")
        config.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        log_message("OCR Engine Ready.")
    else:
        log_message("OCR Engine already loaded, reusing.")
    config.app_phase = "scan_qr"
    log_message(f"=== Problem {config.problem_number} — Scanning QR code ===")


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
        pygame.display.set_mode(
            (info.current_w, info.current_h), pygame.FULLSCREEN | pygame.SCALED
        )
    else:
        pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)


def main(mode):
    config.init_config()
    config.args.mode = mode
    config.running = True
    config.app_phase = "start"
    config.current_step = "firstStep"
    config.transformation_progress = 0.0
    config.green_count = 0
    config.red_count = 0
    config.feedback_state = None
    config.feedback_timer = 0
    config.show_hint = False
    config.is_processing = False
    config.debug_preview = False
    config.nasa_tlx_responses = [None] * 6
    config.ueq_s_responses = [None] * 8
    config.nasa_tlx_current_page = 0
    config.status_msg = "System Ready. Place paper to align ArUco markers."
    config.cap = None
    config.ocr_reader = None
    config.problem_loaded = False
    config.active_vectors = []
    config.target_vector = None
    config.expected_answers = {}
    config.problem_number = 1
    config.problem_ended_early = False
    config._problem_start_str = None
    config._problem_start_time = None

    screen = pygame.display.set_mode(
        (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE
    )
    pygame.display.set_caption("LineAR - Production Projector Space")
    clock = pygame.time.Clock()
    init_session_files()
    ui._init_fonts(min(1.2, config.WINDOW_HEIGHT / 720))

    log_message("Initializing hardware camera capture access...")
    camera_device = None
    for i in range(8):
        dev = f'/dev/video{i}'
        if not os.path.exists(dev):
            continue
        try:
            fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
            buf = bytearray(108)
            fcntl.ioctl(fd, 0x80685600, buf)  # VIDIOC_QUERYCAP
            card = buf[16:48].split(b'\x00')[0].decode()
            device_caps = struct.unpack_from('I', buf, 88)[0]
            os.close(fd)
            if "REDRAGON" in card.upper() and (device_caps & 0x00000001):
                camera_device = dev
                break
        except Exception:
            continue

    if camera_device:
        config.cap = cv2.VideoCapture(camera_device)
        log_message(f"Found REDRAGON camera at {camera_device}")
    else:
        config.cap = cv2.VideoCapture(0)
        log_message("REDRAGON not found, falling back to default camera")
    config.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    config.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not config.cap.isOpened():
        log_message(
            "CRITICAL ERROR: Could not open the system video capture stream.")
    else:
        config.CAM_W = int(config.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        config.CAM_H = int(config.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log_message(f"Camera stream initialized: {config.CAM_W}x{config.CAM_H}")
    log_message("Booting up backend real-time tracking daemon thread pass...")
    threading.Thread(target=paper_tracking_daemon, daemon=True).start()

    # Keep persistent bounding box rects across frames so the event handler can read them
    debug_btn_rect = pygame.Rect(0, 0, 0, 0)
    start_btn_rect = pygame.Rect(0, 0, 0, 0)
    end_btn_rect = pygame.Rect(0, 0, 0, 0)
    done_btn_rect = pygame.Rect(0, 0, 0, 0)
    new_problem_btn_rect = pygame.Rect(0, 0, 0, 0)

    while config.running:
        screen.fill(config.COLOR_BG)

        # Pre-calculate UI geometry maps BEFORE running event checks
        left_rect, center_rect, right_rect = config.get_panel_rects(
            screen.get_width(), screen.get_height()
        )

        # Dynamically determine where the button rect boundaries are located on this frame
        start_btn_w = max(100, min(260, int(center_rect.width * 0.30)))
        start_btn_h = max(28, min(70, int(start_btn_w * 70 / 260)))
        start_btn_rect = pygame.Rect(
            center_rect.x + (center_rect.width - start_btn_w) // 2,
            center_rect.y + (center_rect.height - start_btn_h) // 2,
            start_btn_w,
            start_btn_h,
        )

        end_box_w = max(80, min(160, int(center_rect.width * 0.18)))
        end_box_h = max(22, min(44, int(end_box_w * 44 / 160)))
        end_btn_rect = pygame.Rect(
            center_rect.centerx - end_box_w // 2,
            screen.get_height()
            - config.BOTTOM_BAR_HEIGHT
            + (config.BOTTOM_BAR_HEIGHT - end_box_h) // 2,
            end_box_w,
            end_box_h,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                handle_shutdown(from_button=False)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if config.app_phase == "nasa_tlx":
                    result = handle_nasa_tlx_click(event.pos)
                    if result == "done":
                        config.app_phase = "ueq_s"
                        log_message("NASA-TLX completed, proceeding to UEQ-S.")
                elif config.app_phase == "ueq_s":
                    result = handle_ueq_s_click(event.pos)
                    if result == "submit":
                        save_questionnaire_responses()
                        return_to_launcher()
                elif config.app_phase == "start" and start_btn_rect.collidepoint(
                    event.pos
                ):
                    start_session()
                elif config.app_phase == "done" and new_problem_btn_rect.collidepoint(
                    event.pos
                ):
                    save_problem_results()
                    reset_for_new_problem()
                elif config.app_phase == "done" and done_btn_rect.collidepoint(
                    event.pos
                ):
                    save_problem_results()
                    config.nasa_tlx_current_page = 0
                    config.app_phase = "nasa_tlx"
                    log_message(
                        "Task complete. Starting NASA-TLX questionnaire.")
                elif config.app_phase == "running" and end_btn_rect.collidepoint(
                    event.pos
                ):
                    config.problem_ended_early = True
                    if os.path.exists(config.SESSION_PATH):
                        with open(config.SESSION_PATH) as f:
                            data = json.load(f)
                        data["end_task_pressed"] = True
                        with open(config.SESSION_PATH, "w") as f:
                            json.dump(data, f, indent=2)
                    config.nasa_tlx_current_page = 0
                    config.app_phase = "nasa_tlx"
                    log_message(
                        "Task ended early. Starting NASA-TLX questionnaire.")
                elif getattr(
                    config, "debug_mode", False
                ) and debug_btn_rect.collidepoint(event.pos):
                    advance_debug_state()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    config.debug_mode = not config.debug_mode
                    log_message(f"Debug interface display set to: {
                                config.debug_mode}")
                elif event.key == pygame.K_h:
                    config.debug_hints = not config.debug_hints
                    log_message(f"Debug hints display set to: {
                                config.debug_hints}")
                elif event.key == pygame.K_g and config.debug_mode:
                    debug_simulate_correct()
                elif event.key == pygame.K_r and config.debug_mode:
                    debug_simulate_incorrect()
                elif event.key == pygame.K_n and config.debug_mode:
                    advance_debug_state()
                elif (
                    event.key == pygame.K_p
                    and config.debug_mode
                    and config.args.mode == "highlights"
                ):
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
                        save_problem_results()
                        config.nasa_tlx_current_page = 0
                        config.app_phase = "nasa_tlx"
                        log_message(
                            "Task complete. Starting NASA-TLX questionnaire.")
                    elif config.app_phase == "nasa_tlx":
                        if (
                            config.nasa_tlx_responses[config.nasa_tlx_current_page]
                            is not None
                        ):
                            config.nasa_tlx_current_page += 1
                            if config.nasa_tlx_current_page >= len(
                                config.nasa_tlx_responses
                            ):
                                config.app_phase = "ueq_s"
                                log_message(
                                    "NASA-TLX completed, proceeding to UEQ-S.")
                    elif config.app_phase == "ueq_s":
                        save_questionnaire_responses()
                        return_to_launcher()
                    elif not config.is_processing:
                        config.is_processing = True
                        threading.Thread(
                            target=background_ocr_pipeline, daemon=True
                        ).start()
                elif event.key == pygame.K_F11:
                    config.fullscreen = not config.fullscreen
                    toggle_fullscreen()

            elif event.type == pygame.VIDEORESIZE:
                if not config.fullscreen:
                    screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )
                    ui._init_fonts(min(1.2, event.h / 720))

        if (
            config.app_phase == "scan_qr"
            and not config.problem_loaded
            and not config.is_processing
            and config.latest_frame is not None
        ):
            now_scan = time.time()
            if now_scan - config.last_ocr_time >= 2.0:
                config.last_ocr_time = now_scan
                config.is_processing = True
                threading.Thread(target=scan_qr_from_camera,
                                 daemon=True).start()

        if (
            config.app_phase == "scan_qr"
            and config.problem_loaded
            and not config.is_processing
        ):
            config.app_phase = "running"
            config._problem_start_str = datetime.now().strftime("%H:%M:%S")
            config._problem_start_time = time.time()
            log_message(f"Problem loaded! === Problem {config.problem_number} Started ===")

        if config.app_phase == "running" and config.problem_loaded and not config.is_processing:
            marker_elapsed = time.time() - config.markers_visible_since
            remaining = max(0, 3.0 - marker_elapsed)
            if remaining > 0:
                countdown = int(remaining) + 1
                if not hasattr(config, '_last_countdown') or config._last_countdown != countdown:
                    config._last_countdown = countdown
                    log_message(f"OCR ready in {countdown}s (markers visible for {marker_elapsed:.1f}s)")
            else:
                if not hasattr(config, '_last_countdown') or config._last_countdown != 0:
                    config._last_countdown = 0
                    log_message("Markers stable — OCR eligible")
            if (
                time.time() - config.last_ocr_time >= 1.0
                and marker_elapsed >= 3.0
            ):
                config.last_ocr_time = time.time()
                config.is_processing = True
                threading.Thread(
                    target=background_ocr_pipeline, daemon=True).start()

        if config.app_phase == "running" and config.problem_loaded and not config.is_processing:
            expected = config.expected_answers.get(config.current_step, {})
            regions = config.CROP_REGIONS.get(config.current_step, {})
            if not expected and not regions:
                idx = config.STEP_SEQUENCE.index(config.current_step)
                if idx < len(config.STEP_SEQUENCE) - 1:
                    config.current_step = config.STEP_SEQUENCE[idx + 1]
                    log_message(f"AUTO-ADVANCE: Step has no OCR requirements, advancing to '{config.current_step}'.")

        if config.current_step == "complete":
            if config.transformation_progress < 1.0:
                config.transformation_progress += 0.003
            if config.complete_point_progress < 1.0:
                config.complete_point_progress += 0.003

        config.transformation_progress = max(
            0.0, min(1.0, config.transformation_progress)
        )
        config.complete_point_progress = max(
            0.0, min(1.0, config.complete_point_progress)
        )

        ui.draw_top_bar(screen)
        returned_debug_rect = ui.draw_bottom_bar(screen, center_rect)
        if returned_debug_rect:
            debug_btn_rect = returned_debug_rect

        if config.app_phase == "start":
            center_rect = ui.draw_panels(screen, mode="start")
            start_btn_rect = ui.draw_start_button(screen, center_rect)
        elif config.app_phase == "scan_qr":
            center_rect = ui.draw_panels(screen, mode="scan_qr")
        elif config.app_phase == "done":
            center_rect = ui.draw_panels(screen, mode="done")
            new_problem_btn_rect = ui.draw_new_problem_button(
                screen, center_rect, y_offset=-60
            )
            done_btn_rect = ui.draw_done_button(
                screen, center_rect, y_offset=60)
        elif config.app_phase == "nasa_tlx":
            draw_nasa_tlx(screen)
        elif config.app_phase == "ueq_s":
            draw_ueq_s(screen)
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