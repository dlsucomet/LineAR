import sys
import os
import re
import cv2
import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server.process_paper import process_image, CROP_REGIONS, CAPTURES_DIR

pygame.init()

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
TOP_BAR_HEIGHT = 50
BOTTOM_BAR_HEIGHT = 70
OUTER_GAP = 14
INNER_GAP = 14

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BG = (248, 250, 252)
COLOR_GRID = (226, 232, 240)
COLOR_DARK_GRAY = (100, 100, 100)
COLOR_BLUE = (26, 58, 107)

EXPECTED_ANSWERS = {
    "secondStepLeft":  {"one": ["3"],    "two": ["4", "-1", "-4", "1"]},
    "secondStepRight": {"one": ["-2", "2"],   "two": ["-1", "2"]},
    "thirdStepLeft":   {"one": ["12", "1", "2", "3", "-3"]},
    "thirdStepRight":  {"one": ["2", "-4", "-2", "4"]},
    "fourthStep":      {"one": ["14", "7", "-7"]},
}

STEP_ORDER = [
    "secondStepLeft",
    "secondStepRight",
    "thirdStepLeft",
    "thirdStepRight",
    "fourthStep",
]

STEP_INSTRUCTIONS = {
    "secondStepLeft": {
        "heading": "Vector Addition Property",
        "text": "Linear transformations distribute over scalar values and vector groupings:",
        "equation": "T(u + w) = T(u) + T(w)",
    },
    "secondStepRight": {
        "heading": "Matrix Scaling Step",
        "text": "Progress forward by inputting scalar operations in the next region.",
        "equation": "T(c \u00b7 v) = c \u00b7 T(v)",
    },
    "thirdStepLeft": {
        "heading": "Composition of Transformations",
        "text": "Apply sequential transformations to verify the composition property:",
        "equation": "(T \u2218 S)(v) = T(S(v))",
    },
    "thirdStepRight": {
        "heading": "Kernel Verification",
        "text": "Identify vectors that map to zero under the transformation:",
        "equation": "ker(T) = {v \u2223 T(v) = 0}",
    },
    "fourthStep": {
        "heading": "Final Assembly",
        "text": "Combine all verified properties to complete the proof:",
        "equation": "T(u + w) = T(u) + T(w)",
    },
}

STEP_HINTS = {
    "secondStepLeft": "Step Verified. Progress forward by inputting scalar operations in the next region.",
    "secondStepRight": "Step Verified. Continue to composition verification.",
    "thirdStepLeft": "Step Verified. Move to kernel analysis.",
    "thirdStepRight": "Step Verified. Proceed to final assembly.",
    "fourthStep": "All steps completed! Transformation verified.",
}

STEP_VECTORS = {
    "secondStepLeft":  {"x": 3, "y": -2, "label": "Target Proj"},
}

current_step_index = 0
active_vectors = [
    {"x": 1, "y": 2, "label": "u"},
    {"x": 0, "y": 1, "label": "w"},
]
status_text = "System Active"

camera = None
fullscreen = False
screen = None
hint_timer = 0
hint_message = ""
show_hint = False

font_large = pygame.font.SysFont("segoeui", 24, bold=True)
font_medium = pygame.font.SysFont("segoeui", 18)
font_small = pygame.font.SysFont("segoeui", 15)
font_bold_small = pygame.font.SysFont("segoeui", 15, bold=True)
font_badge = pygame.font.SysFont("segoeui", 13)
font_equation = pygame.font.SysFont("segoeui", 22, bold=True)
font_hint = pygame.font.SysFont("segoeui", 14)

verify_btn_rect = pygame.Rect(0, 0, 160, 44)
exit_btn_rect = pygame.Rect(0, 0, 120, 44)


def get_current_step():
    return STEP_ORDER[current_step_index] if current_step_index < len(STEP_ORDER) else None


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
    fullscreen = not fullscreen
    if fullscreen:
        info = pygame.display.Info()
        WINDOW_WIDTH = info.current_w
        WINDOW_HEIGHT = info.current_h
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
    else:
        WINDOW_WIDTH = 1920
        WINDOW_HEIGHT = 1080
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)


def init_camera():
    global camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        return False
    return True


def capture_frame():
    if camera is None:
        return None
    ret, frame = camera.read()
    if not ret:
        return None
    return cv2.flip(frame, 1)


def validate_answers(recognized):
    step = get_current_step()
    if step is None:
        return "correct"
    expected = EXPECTED_ANSWERS[step]
    for region_name, expected_values in expected.items():
        recognized_values = recognized.get(region_name, [])
        if len(recognized_values) == 0:
            return "incomplete"
        contains_valid = any(
            any(re.search(re.escape(val), ocr_str) for ocr_str in recognized_values)
            for val in expected_values
        )
        if not contains_valid:
            return "wrong"
    return "correct"


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

    badge_text = font_badge.render(status_text, True, COLOR_WHITE)
    pad = 10
    badge_w = badge_text.get_width() + pad * 2
    badge_h = 26
    badge_x = W - badge_w - 16
    badge_y = (TOP_BAR_HEIGHT - badge_h) // 2
    pygame.draw.rect(surface, COLOR_BLACK, (badge_x, badge_y, badge_w, badge_h))
    pygame.draw.rect(surface, COLOR_WHITE, (badge_x, badge_y, badge_w, badge_h), 1)
    surface.blit(badge_text, (badge_x + pad, badge_y + (badge_h - badge_text.get_height()) // 2))


def draw_bottom_bar(surface, buttons):
    W = surface.get_width()
    H = surface.get_height()
    bar_rect = pygame.Rect(0, H - BOTTOM_BAR_HEIGHT, W, BOTTOM_BAR_HEIGHT)
    pygame.draw.rect(surface, COLOR_BG, bar_rect)

    cx = W // 2
    by = H - BOTTOM_BAR_HEIGHT + (BOTTOM_BAR_HEIGHT - 44) // 2

    verify_btn_rect.topleft = (cx - 160, by)
    exit_btn_rect.topleft = (cx + 20, by)

    for rect, is_hover, label, fill in buttons:
        if fill:
            color = (60, 60, 60) if is_hover else COLOR_BLUE
            pygame.draw.rect(surface, color, rect, border_radius=4)
            txt = font_medium.render(label, True, COLOR_WHITE)
        else:
            color = (200, 200, 200) if is_hover else COLOR_WHITE
            pygame.draw.rect(surface, color, rect, border_radius=4)
            pygame.draw.rect(surface, COLOR_BLUE, rect, 2, border_radius=4)
            txt = font_medium.render(label, True, COLOR_BLACK)
        txt_rect = txt.get_rect(center=rect.center)
        surface.blit(txt, txt_rect)


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
    step = get_current_step()
    if step:
        draw_ar_overlay(surface, paper_area, step)

    draw_instruction_panel(surface, right_rect)


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

    pygame.draw.line(surface, COLOR_BLACK, (area.x, oy), (area.x + area.width, oy), 2)
    pygame.draw.line(surface, COLOR_BLACK, (ox, area.y), (ox, area.y + area.height), 2)

    clip_rect = surface.get_clip()
    surface.set_clip(area)

    for vec in active_vectors:
        tx = ox + int(vec["x"] * scale)
        ty = oy - int(vec["y"] * scale)

        if tx < area.x or tx > area.x + area.width or ty < area.y or ty > area.y + area.height:
            continue

        pygame.draw.line(surface, COLOR_BLACK, (ox, oy), (tx, ty), 4)
        pygame.draw.circle(surface, COLOR_BLACK, (tx, ty), 5)

        label = f'{vec["label"]} ({vec["x"]},{vec["y"]})'
        txt = font_bold_small.render(label, True, COLOR_BLACK)
        surface.blit(txt, (tx + 8, ty - 4))

    surface.set_clip(clip_rect)


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

        draw_dashed_rect(surface, COLOR_BLACK, (rx, ry, rw, rh), 3, dash_len, gap_len)


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


def draw_instruction_panel(surface, area):
    step = get_current_step()
    if step is None:
        txt = font_large.render("All steps completed!", True, COLOR_BLACK)
        txt_rect = txt.get_rect(center=area.center)
        surface.blit(txt, txt_rect)
        return

    instr = STEP_INSTRUCTIONS[step]
    pad = 20
    content_x = area.x + pad
    content_y = area.y + pad
    max_w = area.width - pad * 2

    heading = font_large.render(instr["heading"], True, COLOR_BLACK)
    surface.blit(heading, (content_x, content_y))
    content_y += 40

    text_lines = wrap_text(instr["text"], font_small, max_w)
    for line in text_lines:
        rendered = font_small.render(line, True, COLOR_BLACK)
        surface.blit(rendered, (content_x, content_y))
        content_y += 22
    content_y += 10

    eq_card_rect = pygame.Rect(content_x, content_y, max_w, 50)
    pygame.draw.rect(surface, COLOR_WHITE, eq_card_rect)
    pygame.draw.rect(surface, COLOR_BLACK, eq_card_rect, 2)
    eq = font_equation.render(instr["equation"], True, COLOR_BLACK)
    eq_rect = eq.get_rect(center=eq_card_rect.center)
    surface.blit(eq, eq_rect)
    content_y += 70

    if show_hint:
        hint_rect = pygame.Rect(content_x, content_y, max_w, 60)
        pygame.draw.rect(surface, COLOR_WHITE, hint_rect)
        pygame.draw.rect(surface, COLOR_BLACK, hint_rect, 2, border_radius=4)
        hint_lines = wrap_text(hint_message, font_hint, max_w - 20)
        hy = hint_rect.y + 8
        for line in hint_lines:
            rendered = font_hint.render(line, True, COLOR_BLACK)
            surface.blit(rendered, (hint_rect.x + 10, hy))
            hy += 18


def handle_verify():
    global status_text, current_step_index, active_vectors, hint_timer, hint_message, show_hint

    step = get_current_step()
    if step is None:
        status_text = "All steps completed!"
        return

    status_text = "Analyzing handwritten elements..."

    frame = capture_frame()
    if frame is None:
        status_text = "Error: No camera frame"
        return

    timestamp = int(pygame.time.get_ticks())
    frame_path = os.path.join(CAPTURES_DIR, f"hud-frame-{timestamp}.png")
    cv2.imwrite(frame_path, frame)

    recognized, proc_status, _ = process_image(frame, step, debug_prefix=f"{step}-{timestamp}")

    if recognized is None:
        status_text = f"Paper not found: {proc_status}"
        return

    state = validate_answers(recognized)
    if state == "correct":
        if step in STEP_VECTORS:
            active_vectors.append(STEP_VECTORS[step].copy())
        current_step_index += 1

        hint_message = STEP_HINTS.get(step, "Step Verified!")
        hint_timer = 180
        show_hint = True

        next_step = get_current_step()
        if next_step is None:
            status_text = "Sheet completed! All steps validated."
        else:
            status_text = "Step Validated! Correct."
    elif state == "incomplete":
        status_text = "No digits recognized in region"
    else:
        status_text = "Incorrect answer, try again"


def handle_exit():
    pygame.quit()
    if camera is not None:
        camera.release()
    sys.exit()


def main():
    global status_text, camera, hint_timer, show_hint, screen, WINDOW_WIDTH, WINDOW_HEIGHT

    os.makedirs(CAPTURES_DIR, exist_ok=True)

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("LineAR - Projector Optimization Workspace")
    clock = pygame.time.Clock()

    if not init_camera():
        status_text = "Error: No camera detected"

    running = True
    while running:
        if hint_timer > 0:
            hint_timer -= 1
            if hint_timer == 0:
                show_hint = False

        mx, my = pygame.mouse.get_pos()
        v_hover = verify_btn_rect.collidepoint(mx, my)
        e_hover = exit_btn_rect.collidepoint(mx, my)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                handle_exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if verify_btn_rect.collidepoint(event.pos):
                    handle_verify()
                elif exit_btn_rect.collidepoint(event.pos):
                    handle_exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    toggle_fullscreen()
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    handle_verify()
                elif event.key == pygame.K_ESCAPE:
                    handle_exit()

        screen.fill(COLOR_BG)
        draw_top_bar(screen)
        draw_panels(screen)
        draw_bottom_bar(screen, [
            (verify_btn_rect, v_hover, "VERIFY STEPS", True),
            (exit_btn_rect, e_hover, "EXIT", False),
        ])
        pygame.display.flip()
        clock.tick(30)

    handle_exit()


if __name__ == "__main__":
    main()
