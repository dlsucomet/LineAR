import re
import pygame
import config
from pipeline import transform_to_projection_space

pygame.font.init()
font_large = pygame.font.SysFont("segoeui", 24, bold=True)
font_medium = pygame.font.SysFont("segoeui", 18)
font_body = pygame.font.SysFont("segoeui", 15)
font_bold = pygame.font.SysFont("segoeui", 15, bold=True)
font_equation = pygame.font.SysFont("segoeui", 15, bold=True)

# Step-by-step guidance dictionary configuration
STEP_GUIDANCE = {
    "firstStepLeft": {
        "title": "Linear Combination Setup",
        "desc": "Express the input vector as a linear combination of the given basis vectors.",
        "math": "L(c1*v1 + c2*v2) = c1*L(v1) + c2*L(v2)"
    },
    "firstStepRight": {
        "title": "Transformation Property Expansion",
        "desc": "Substitute the known transformed vector definitions into your linear combination equation.",
        "math": "c1 * [x1 / y1] + c2 * [x2 / y2]"
    },
    "secondStepLeft": {
        "title": "Scalar Multiplication",
        "desc": "Distribute the first scalar coefficient into the left vector component elements.",
        "math": "a * [b / c] = [a * b / a * c]"
    },
    "secondStepRight": {
        "title": "Scalar Multiplication",
        "desc": "Distribute the second scalar coefficient into the right vector component elements.",
        "math": "a * [b / c] = [a * b / a * c]"
    },
    "thirdStep": {
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

def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def draw_top_bar(surface):
    bar_rect = pygame.Rect(0, 0, surface.get_width(), config.TOP_BAR_HEIGHT)
    pygame.draw.rect(surface, config.COLOR_BLUE, bar_rect)
    text = font_large.render("Place paper on the designated projection area", True, config.COLOR_WHITE)
    surface.blit(text, text.get_rect(center=(surface.get_width() // 2, config.TOP_BAR_HEIGHT // 2)))

def draw_bottom_bar(surface, center_rect=None):
    H = surface.get_height()
    bar_rect = pygame.Rect(0, H - config.BOTTOM_BAR_HEIGHT, surface.get_width(), config.BOTTOM_BAR_HEIGHT)
    pygame.draw.rect(surface, config.COLOR_BG, bar_rect)

    msg = config.status_msg
    if msg:
        wrap_right = bar_rect.right - config.OUTER_GAP
        if center_rect is not None:
            end_btn_w = max(80, min(160, int(center_rect.width * 0.18)))
            end_btn_x = center_rect.centerx - end_btn_w // 2
            wrap_right = end_btn_x - config.OUTER_GAP
        avail_w = wrap_right - config.OUTER_GAP
        if avail_w > 20:
            full_text = f"CONSOLE LOG: {msg}"
            lines = wrap_text(full_text, font_bold, avail_w)
            show_hints = getattr(config, "debug_mode", False) and getattr(config, "debug_hints", False)
            max_lines = 2 if show_hints else 3
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                last = lines[-1]
                if len(last) > 3:
                    lines[-1] = last[:-3] + "..."
            line_h = font_bold.get_height() + 2
            total_h = len(lines) * line_h
            start_y = bar_rect.y + (bar_rect.height - total_h) // 2
            for i, line in enumerate(lines):
                surf = font_bold.render(line, True, config.COLOR_TEXT)
                surface.blit(surf, (config.OUTER_GAP, start_y + i * line_h))

    if getattr(config, "debug_mode", False) and getattr(config, "debug_hints", False):
        hints = "[G] Correct | [R] Incorrect | [H] Hints"
        hint_surf = font_bold.render(hints, True, config.COLOR_TEXT)
        surface.blit(hint_surf, (config.OUTER_GAP, bar_rect.bottom - hint_surf.get_height() - 6))

    if getattr(config, "debug_mode", False):
        return draw_debug_button(surface)
    return None

def draw_cartesian_plane(surface, area):
    pygame.draw.rect(surface, config.COLOR_WHITE, area)
    ox, oy = area.x + area.width // 2, area.y + area.height // 2
    scale = max(1, int(min(area.width, area.height) / 14))
    grid_surf = pygame.Surface((area.width, area.height), pygame.SRCALPHA)
    for x in range(ox % scale, area.x + area.width, scale):
        local_x = x - area.x
        pygame.draw.line(grid_surf, (*config.COLOR_GRID, 180), (local_x, 0), (local_x, area.height))
    for y in range(oy % scale, area.y + area.height, scale):
        local_y = y - area.y
        pygame.draw.line(grid_surf, (*config.COLOR_GRID, 180), (0, local_y), (area.width, local_y))
    surface.blit(grid_surf, (area.x, area.y))
    pygame.draw.line(surface, config.COLOR_TEXT, (area.x, oy), (area.x + area.width, oy), 2)
    pygame.draw.line(surface, config.COLOR_TEXT, (ox, area.y), (ox, area.y + area.height), 2)
    
    clip_rect = surface.get_clip()
    surface.set_clip(area)
    for vec in config.active_vectors:
        tx, ty = ox + int(vec["x"] * scale), oy - int(vec["y"] * scale)
        if area.x <= tx <= area.x + area.width and area.y <= ty <= area.y + area.height:
            pygame.draw.line(surface, config.COLOR_TEXT, (ox, oy), (tx, ty), 4)
            pygame.draw.circle(surface, config.COLOR_TEXT, (tx, ty), 5)
            surface.blit(font_bold.render(f'{vec["label"]} ({vec["x"]},{vec["y"]})', True, config.COLOR_TEXT), (tx + 8, ty - 4))
    surface.set_clip(clip_rect)

def draw_instruction_panel(surface, area):
    pad = 20
    cx, cy, mw = area.x + pad, area.y + pad, area.width - pad * 2
    step_info = STEP_GUIDANCE.get(config.current_step, STEP_GUIDANCE["complete"])

    default_panel_h = 572
    scale_y = max(0.5, area.height / default_panel_h)

    for line in wrap_text(step_info["title"], font_large, mw):
        surface.blit(font_large.render(line, True, config.COLOR_TEXT), (cx, cy))
        cy += int(36 * scale_y)
    cy += int(4 * scale_y)
    for line in wrap_text(step_info["desc"], font_body, mw):
        surface.blit(font_body.render(line, True, config.COLOR_TEXT), (cx, cy))
        cy += int(22 * scale_y)
    cy += int(10 * scale_y)

    bottom_limit = area.y + area.height - pad
    eq_card_h = max(20, min(int(70 * scale_y), bottom_limit - cy))

    if eq_card_h > 20:
        eq_rect = pygame.Rect(cx, cy, mw, eq_card_h)
        pygame.draw.rect(surface, config.COLOR_WHITE, eq_rect)
        pygame.draw.rect(surface, config.COLOR_TEXT, eq_rect, 2)

        raw_math = step_info["math"]

        parts = []
        for seg in re.split(r'(\[[^\[\]]*\])', raw_math):
            if not seg:
                continue
            if seg.startswith("[") and seg.endswith("]"):
                inner = seg[1:-1]
                rows = [r.strip() for r in inner.split("/")]
                parts.append(("vector", rows))
            else:
                parts.append(("text", seg))

        eq_max_w = mw - 8
        eq_font_size = 15
        for size in range(15, 8, -2):
            test_font = pygame.font.SysFont("segoeui", size, bold=True)
            total_w = 0
            for ptype, pcontent in parts:
                if ptype == "text":
                    total_w += test_font.size(pcontent)[0]
                else:
                    row_widths = [test_font.size(r)[0] for r in pcontent]
                    total_w += (max(row_widths) if row_widths else 0) + 24
            if total_w <= eq_max_w:
                eq_font_size = size
                break

        use_font = pygame.font.SysFont("segoeui", eq_font_size, bold=True)
        row_h = use_font.get_height()
        center_y = eq_rect.centery
        current_x = eq_rect.x + 15
        line_spacing = 4

        for i, (ptype, pcontent) in enumerate(parts):
            if ptype == "text":
                t_surf = use_font.render(pcontent, True, config.COLOR_TEXT)
                surface.blit(t_surf, (current_x, center_y - t_surf.get_height() // 2))
                current_x += t_surf.get_width()
            else:
                rows = pcontent
                row_surfaces = [use_font.render(r, True, config.COLOR_TEXT) for r in rows]
                max_row_w = max(s.get_width() for s in row_surfaces)
                total_h = len(rows) * row_h + (len(rows) - 1) * line_spacing
                bracket_top = center_y - total_h // 2
                bracket_bottom = bracket_top + total_h

                pygame.draw.line(surface, config.COLOR_TEXT, (current_x, bracket_top), (current_x, bracket_bottom), 2)
                pygame.draw.line(surface, config.COLOR_TEXT, (current_x, bracket_top), (current_x + 4, bracket_top), 2)
                pygame.draw.line(surface, config.COLOR_TEXT, (current_x, bracket_bottom), (current_x + 4, bracket_bottom), 2)

                content_x = current_x + 6
                for j, surf in enumerate(row_surfaces):
                    row_y = bracket_top + j * (row_h + line_spacing)
                    surface.blit(surf, (content_x + (max_row_w - surf.get_width()) // 2, row_y))

                right_x = content_x + max_row_w + 6
                pygame.draw.line(surface, config.COLOR_TEXT, (right_x, bracket_top), (right_x, bracket_bottom), 2)
                pygame.draw.line(surface, config.COLOR_TEXT, (right_x, bracket_top), (right_x - 4, bracket_top), 2)
                pygame.draw.line(surface, config.COLOR_TEXT, (right_x, bracket_bottom), (right_x - 4, bracket_bottom), 2)

                current_x = right_x

def draw_panels(surface, mode="running"):
    left, center, right = config.get_panel_rects(surface.get_width(), surface.get_height())
    for rect in [left, center, right]:
        pygame.draw.rect(surface, config.COLOR_WHITE, rect)
        pygame.draw.rect(surface, config.COLOR_BLUE, rect, 3)

    if mode == "start":
        return center

    draw_cartesian_plane(surface, left)
    pygame.draw.rect(surface, config.COLOR_BLUE, left, 3)

    if mode == "done":
        draw_instruction_panel(surface, right)
        return center

    if config.args.mode == "highlights" and config.current_step in config.PROJECTION_REGIONS:
        with config.shared_frame_lock:
            tracking = config.paper_detected
        if tracking:
            for _, box in config.PROJECTION_REGIONS[config.current_step].items():
                top_left = transform_to_projection_space(box["left"], box["top"], center)
                bottom_right = transform_to_projection_space(box["left"] + box["width"], box["top"] + box["height"], center)
                if top_left and bottom_right:
                    pygame.draw.rect(surface, config.COLOR_AXIS, (top_left[0], top_left[1], bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]), 2)
        else:
            msg = font_bold.render("[ Align ArUco Markers to Project Guides ]", True, config.COLOR_TEXT)
            surface.blit(msg, msg.get_rect(center=center.center))

    if config.feedback_timer > 0 and config.feedback_state == "green":
        green = (34, 197, 94)
        pygame.draw.rect(surface, green, center, 8)
        label = font_large.render("CORRECT", True, green)
        surface.blit(label, label.get_rect(center=(center.centerx, center.y + 30)))
    elif config.feedback_timer > 0 and config.feedback_state == "red":
        red = (239, 68, 68)
        pygame.draw.rect(surface, red, center, 8)
        label = font_large.render("INCORRECT", True, red)
        surface.blit(label, label.get_rect(center=(center.centerx, center.y + 30)))

    draw_instruction_panel(surface, right)
    return center

def draw_start_button(surface, center_rect):
    btn_w = max(100, min(260, int(center_rect.width * 0.30)))
    btn_h = max(28, min(70, int(btn_w * 70 / 260)))
    btn_rect = pygame.Rect(
        center_rect.x + (center_rect.width - btn_w) // 2,
        center_rect.y + (center_rect.height - btn_h) // 2, 
        btn_w, 
        btn_h
    )
    mx, my = pygame.mouse.get_pos()
    hovered = btn_rect.collidepoint(mx, my)
    if hovered:
        pygame.draw.rect(surface, config.COLOR_WHITE, btn_rect, border_radius=12)
        pygame.draw.rect(surface, config.COLOR_BLUE, btn_rect, 3, border_radius=12)
        text = font_large.render("START", True, config.COLOR_BLUE)
    else:
        pygame.draw.rect(surface, config.COLOR_BLUE, btn_rect, border_radius=12)
        text = font_large.render("START", True, config.COLOR_WHITE)
    text_rect = text.get_rect(center=btn_rect.center)
    surface.blit(text, text_rect)
    return btn_rect


def draw_end_task_button(surface, center_rect):
    btn_w = max(80, min(160, int(center_rect.width * 0.18)))
    btn_h = max(22, min(44, int(btn_w * 44 / 160)))
    H = surface.get_height()
    rect = pygame.Rect(center_rect.centerx - btn_w // 2, H - config.BOTTOM_BAR_HEIGHT + (config.BOTTOM_BAR_HEIGHT - btn_h) // 2, btn_w, btn_h)
    hovered = rect.collidepoint(pygame.mouse.get_pos())
    pygame.draw.rect(surface, config.COLOR_WHITE if hovered else config.COLOR_BLUE, rect, border_radius=6)
    pygame.draw.rect(surface, config.COLOR_BLUE, rect, 3, border_radius=6)
    t = font_medium.render("End Task", True, config.COLOR_BLUE if hovered else config.COLOR_WHITE)
    surface.blit(t, t.get_rect(center=rect.center))
    return rect

def draw_done_button(surface, center_rect):
    btn_w = max(100, min(260, int(center_rect.width * 0.30)))
    btn_h = max(28, min(70, int(btn_w * 70 / 260)))
    btn_rect = pygame.Rect(
        center_rect.x + (center_rect.width - btn_w) // 2,
        center_rect.y + (center_rect.height - btn_h) // 2,
        btn_w,
        btn_h
    )
    mx, my = pygame.mouse.get_pos()
    hovered = btn_rect.collidepoint(mx, my)
    green = (34, 197, 94)
    if hovered:
        pygame.draw.rect(surface, config.COLOR_WHITE, btn_rect, border_radius=12)
        pygame.draw.rect(surface, green, btn_rect, 3, border_radius=12)
        text = font_large.render("DONE", True, green)
    else:
        pygame.draw.rect(surface, green, btn_rect, border_radius=12)
        text = font_large.render("DONE", True, config.COLOR_WHITE)
    text_rect = text.get_rect(center=btn_rect.center)
    surface.blit(text, text_rect)
    return btn_rect

def draw_debug_button(surface):
    H = surface.get_height()
    W = surface.get_width()
    btn_w, btn_h = 120, 30
    rect = pygame.Rect(W - btn_w - config.OUTER_GAP, H - config.BOTTOM_BAR_HEIGHT + (config.BOTTOM_BAR_HEIGHT - btn_h) // 2, btn_w, btn_h)
    hovered = rect.collidepoint(pygame.mouse.get_pos())
    pygame.draw.rect(surface, config.COLOR_WHITE if hovered else config.COLOR_BLUE, rect, border_radius=4)
    pygame.draw.rect(surface, config.COLOR_BLUE, rect, 2, border_radius=4)
    t = font_bold.render("DEBUG: Next Step", True, config.COLOR_BLUE if hovered else config.COLOR_WHITE)
    surface.blit(t, t.get_rect(center=rect.center))
    return rect