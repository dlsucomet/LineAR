import re
import pygame
import config
import math 
from pipeline import transform_to_projection_space

font_large = None
font_medium = None
font_body = None
font_bold = None
font_equation = None

def _init_fonts():
    global font_large, font_medium, font_body, font_bold, font_equation
    pygame.font.init()
    font_large = pygame.font.SysFont("segoeui", 24, bold=True)
    font_medium = pygame.font.SysFont("segoeui", 18)
    font_body = pygame.font.SysFont("segoeui", 15)
    font_bold = pygame.font.SysFont("segoeui", 15, bold=True)
    font_equation = pygame.font.SysFont("segoeui", 15, bold=True)

# Step-by-step guidance dictionary configuration
STEP_GUIDANCE = {
    "firstStep": {
        "title": "Linear Combination Setup",
        "desc": "Express the input vector as a linear combination of the given basis vectors.",
        "math": "L(c1*v1 + c2*v2) = c1*L(v1) + c2*L(v2)"
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
    if config.app_phase == "start":
        text = font_large.render("Use the pen to click Start", True, config.COLOR_WHITE)
    elif config.app_phase == "scan_qr":
        text = font_large.render("Show the QR code side of your paper to the camera", True, config.COLOR_WHITE)
    elif config.app_phase == "qr_confirm":
        text = font_large.render("Problem loaded successfully!", True, config.COLOR_WHITE)
    elif config.app_phase == "flip_prompt":
        text = font_large.render("Please flip your paper to the problem side", True, config.COLOR_WHITE)
    elif config.app_phase == "done":
        text = font_large.render("Use the pen to click Done", True, config.COLOR_WHITE)
    elif config.app_phase == "nasa_tlx":
        page = config.nasa_tlx_current_page + 1
        total = 6
        title = f"NASA-TLX ({page}/{total})"
        text = font_large.render(f"Questionnaire: {title} — Use the pen to interact", True, config.COLOR_WHITE)
    elif config.app_phase == "ueq_s":
        text = font_large.render("Questionnaire: UEQ-S — Use the pen to interact", True, config.COLOR_WHITE)
    else:
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
        hints = "[G] Correct | [R] Incorrect"
        if getattr(config.args, "mode", "") == "highlights":
            hints += " | [P] Preview"
        hints += " | [H] Hints"
        hint_surf = font_bold.render(hints, True, config.COLOR_TEXT)
        surface.blit(hint_surf, (config.OUTER_GAP, bar_rect.bottom - hint_surf.get_height() - 6))

    if getattr(config, "debug_mode", False):
        return draw_debug_button(surface)
    return None

def draw_cartesian_plane(surface, area):
    pygame.draw.rect(surface, config.COLOR_WHITE, area)
    
    ox, oy = area.x + area.width // 2, area.y + area.height // 2
    
    if getattr(config, "feedback_timer", 0) > 0 and getattr(config, "feedback_state", "") == "green":
        amplitude = 25 
        frequency = 0.1
        bounce_offset = int(amplitude * math.sin(config.feedback_timer * frequency) * (config.feedback_timer / 60.0))
        ox += bounce_offset
        oy -= bounce_offset  
    step_order = config.STEP_SEQUENCE
    current_idx = step_order.index(config.current_step) if config.current_step in step_order else 0
    scale = max(1, int(min(area.width, area.height) / 14))

    max_extent = 0
    for vec in config.active_vectors:
        vec_idx = step_order.index(vec.get("show_at", "firstStep"))
        if vec_idx > current_idx:
            continue
        max_extent = max(max_extent, abs(vec["x"]), abs(vec["y"]))
    if config.current_step == "complete" and config.target_vector:
        tx, ty = config.target_vector
        max_extent = max(max_extent, abs(tx), abs(ty))
    if max_extent > 0:
        half_area = min(area.width, area.height) / 2
        fit_scale = max(1, int(half_area / (max_extent + 5)))
        scale = min(scale, fit_scale)

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

    v_vec = next((v for v in config.active_vectors if v.get("type") == "point" and v["label"] == "v"), None)
    v_px = ox + int(v_vec["x"] * scale) if v_vec else ox
    v_py = oy - int(v_vec["y"] * scale) if v_vec else oy

    for vec in config.active_vectors:
        vec_idx = step_order.index(vec.get("show_at", "firstStep"))
        if vec_idx > current_idx:
            continue

        if vec.get("type") == "point":
            if vec["label"] == "v":
                color = config.COLOR_POINT_LIGHT
                tx, ty = ox + int(vec["x"] * scale), oy - int(vec["y"] * scale)
            elif vec["label"] == "L(v)":
                prog = getattr(config, "transformation_progress", 0.0)
                final_tx = ox + int(vec["x"] * scale)
                final_ty = oy - int(vec["y"] * scale)
                tx = int(v_px + (final_tx - v_px) * prog)
                ty = int(v_py + (final_ty - v_py) * prog)
                color = config.COLOR_POINT_DARK
            else:
                color = config.COLOR_TEXT
                tx, ty = ox + int(vec["x"] * scale), oy - int(vec["y"] * scale)
            if area.x <= tx <= area.x + area.width and area.y <= ty <= area.y + area.height:
                pygame.draw.circle(surface, color, (tx, ty), 6)
                surface.blit(font_bold.render(f'{vec["label"]} ({vec["x"]},{vec["y"]})', True, color), (tx + 16, ty - 4))
        else:
            tx, ty = ox + int(vec["x"] * scale), oy - int(vec["y"] * scale)
            if area.x <= tx <= area.x + area.width and area.y <= ty <= area.y + area.height:
                pygame.draw.line(surface, config.COLOR_TEXT, (ox, oy), (tx, ty), 4)
                angle = math.atan2(oy - ty, tx - ox)
                arrow_len = 14
                arrow_w = 6
                base_lx = tx - arrow_len * math.cos(angle) + arrow_w * math.sin(angle)
                base_ly = ty + arrow_len * math.sin(angle) + arrow_w * math.cos(angle)
                base_rx = tx - arrow_len * math.cos(angle) - arrow_w * math.sin(angle)
                base_ry = ty + arrow_len * math.sin(angle) - arrow_w * math.cos(angle)
                pygame.draw.polygon(surface, config.COLOR_TEXT, [
                    (tx, ty),
                    (int(base_lx), int(base_ly)),
                    (int(base_rx), int(base_ry))
                ])
                surface.blit(font_bold.render(f'{vec["label"]} ({vec["x"]},{vec["y"]})', True, config.COLOR_TEXT), (tx + 16, ty - 4))
    surface.set_clip(clip_rect)


def draw_instruction_panel(surface, area):
    pad = 20
    cx, cy, mw = area.x + pad, area.y + pad, area.width - pad * 2
    step_info = STEP_GUIDANCE.get(config.current_step, STEP_GUIDANCE["complete"])
    if config.current_step == "complete" and config.target_vector:
        tx, ty = config.target_vector
        step_info = dict(step_info, math=f"L(v) = [{tx} / {ty}]")

    default_panel_h = 572
    scale_y = max(0.5, area.height / default_panel_h)

    if config.problem_loaded and getattr(config, "active_vectors", None) and len(config.active_vectors) >= 3:
        u_vec = config.active_vectors[0]
        w_vec = config.active_vectors[1]
        v_vec = config.active_vectors[2]
        lu = config.expected_answers.get("secondStepLeft", ["?", "?"])
        lw = config.expected_answers.get("thirdStepLeft", ["?", "?"])

        header_surf = font_large.render("Problem:", True, config.COLOR_TEXT)
        surface.blit(header_surf, (cx, cy))
        cy += int(36 * scale_y)

        pf = font_medium
        pcol = config.COLOR_TEXT

        def _draw_vvec(surf, x, y, vals):
            strs = [str(vals[0]), str(vals[1])]
            rh = pf.get_height()
            mw_v = max(pf.size(s)[0] for s in strs)
            gap = 2
            total_h = 2 * rh + gap
            bx = x
            pygame.draw.line(surf, pcol, (bx, y), (bx, y + total_h), 2)
            pygame.draw.line(surf, pcol, (bx, y), (bx + 4, y), 2)
            pygame.draw.line(surf, pcol, (bx, y + total_h), (bx + 4, y + total_h), 2)
            tx = bx + 6
            for i, s in enumerate(strs):
                ts = pf.render(s, True, pcol)
                surf.blit(ts, (tx + (mw_v - ts.get_width()) // 2, y + i * (rh + gap)))
            rx = tx + mw_v + 6
            pygame.draw.line(surf, pcol, (rx, y), (rx, y + total_h), 2)
            pygame.draw.line(surf, pcol, (rx, y), (rx - 4, y), 2)
            pygame.draw.line(surf, pcol, (rx, y + total_h), (rx - 4, y + total_h), 2)
            return rx

        def _draw_paren(surf, x, y, h, side, width=2):
            depth = max(6, int(h * 0.25))
            n = 16
            points = []
            for i in range(n + 1):
                t = i / n
                py = y + t * h
                x_off = depth * math.sin(t * math.pi)
                if side == "left":
                    points.append((x - x_off, py))
                else:
                    points.append((x + x_off, py))
            pygame.draw.lines(surf, pcol, False, points, width)
            return depth

        def _draw_text(surf, x, y, txt, center_h=None):
            ts = pf.render(txt, True, pcol)
            draw_y = y
            if center_h is not None:
                draw_y = y + (center_h - ts.get_height()) // 2
            surf.blit(ts, (x, draw_y))
            return x + ts.get_width()

        vx = cx
        vy = cy
        vx = _draw_text(surface, vx, vy, "Let L: R\u00b2 \u2192 R\u00b2 be a linear transformation such that")
        cy += int(24 * scale_y)

        vec_h = 2 * pf.get_height() + 2

        vx = cx
        vy = cy
        vx = _draw_text(surface, vx, vy, "L", center_h=vec_h)
        vx += 16
        _draw_paren(surface, vx, vy, vec_h, "left")
        vx = _draw_vvec(surface, vx, vy, [u_vec['x'], u_vec['y']]) + 2
        vx += _draw_paren(surface, vx, vy, vec_h, "right")
        vx += 16
        vx = _draw_text(surface, vx, vy, " = ", center_h=vec_h)
        _draw_paren(surface, vx, vy, vec_h, "left")
        vx = _draw_vvec(surface, vx, vy, [lu[0], lu[1]]) + 2
        vx += _draw_paren(surface, vx, vy, vec_h, "right")
        vx += 16
        vx = _draw_text(surface, vx, vy, ",  L", center_h=vec_h)
        vx += 16
        _draw_paren(surface, vx, vy, vec_h, "left")
        vx = _draw_vvec(surface, vx, vy, [w_vec['x'], w_vec['y']]) + 2
        vx += _draw_paren(surface, vx, vy, vec_h, "right")
        vx += 16
        vx = _draw_text(surface, vx, vy, " = ", center_h=vec_h)
        _draw_paren(surface, vx, vy, vec_h, "left")
        vx = _draw_vvec(surface, vx, vy, [lw[0], lw[1]]) + 2
        vx += _draw_paren(surface, vx, vy, vec_h, "right")
        cy += int(50 * scale_y)

        vx = cx
        vy = cy
        vx = _draw_text(surface, vx, vy, "Find L", center_h=vec_h)
        vx += 16
        _draw_paren(surface, vx, vy, vec_h, "left")
        vx = _draw_vvec(surface, vx, vy, [v_vec['x'], v_vec['y']]) + 2
        vx += _draw_paren(surface, vx, vy, vec_h, "right")
        vx += 16
        vx = _draw_text(surface, vx, vy, ".", center_h=vec_h)
        cy += int(50 * scale_y)

        sep_y = int(cy)
        pygame.draw.line(surface, config.COLOR_GRID, (cx, sep_y), (cx + mw, sep_y), 1)
        cy += int(10 * scale_y)

    for line in wrap_text(step_info["title"], font_large, mw):
        surface.blit(font_large.render(line, True, config.COLOR_TEXT), (cx, cy))
        cy += int(36 * scale_y)
    cy += int(4 * scale_y)
    for line in wrap_text(step_info["desc"], font_body, mw):
        surface.blit(font_body.render(line, True, config.COLOR_TEXT), (cx, cy))
        cy += int(22 * scale_y)
    cy += int(10 * scale_y)

    bottom_limit = area.y + area.height - pad
    legend_reserve = 0
    if getattr(config.args, "mode", "") == "highlights":
        legend_reserve = 110
        bottom_limit -= legend_reserve
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

    if getattr(config.args, "mode", "") == "highlights":
        legend_items = [
            (config.COLOR_BLUE,   "copy"),
            ((34, 197, 94),       "correct"),
            ((239, 68, 68),       "incorrect"),
        ]
        sq = 16
        line_h = 32
        total_legend_h = len(legend_items) * line_h
        legend_x = cx
        legend_y = area.bottom - pad - total_legend_h
        for color, label in legend_items:
            pygame.draw.rect(surface, color, (legend_x, legend_y, sq, sq))
            txt = font_large.render(label, True, color)
            surface.blit(txt, (legend_x + sq + 6, legend_y - 2))
            legend_y += line_h

def draw_projection_boxes(surface, center):
    with config.shared_frame_lock:
        tracking = config.paper_detected
        track_mat = config.tracking_matrix

    highlight_step = config.current_step
    if config.feedback_timer > 0 and config.feedback_step:
        highlight_step = config.feedback_step

    if highlight_step in config.PROJECTION_REGIONS:
        if tracking:
            if config.args.mode == "highlights":
                for _, box in config.PROJECTION_REGIONS[highlight_step].items():
                    tl = transform_to_projection_space(box["left"], box["top"], center, track_mat)
                    br = transform_to_projection_space(box["left"] + box["width"], box["top"] + box["height"], center, track_mat)
                    if tl and br:
                        pygame.draw.rect(surface, config.COLOR_BLUE, (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]))
        else:
            msg = font_bold.render("[ Align ArUco Markers to Project Guides ]", True, config.COLOR_TEXT)
            surface.blit(msg, msg.get_rect(center=center.center))

    if config.feedback_timer > 0 and config.feedback_step and config.feedback_step in config.CROP_REGIONS:
        if tracking:
            feedback_color = (34, 197, 94) if config.feedback_state == "green" else (239, 68, 68)
            for _, box in config.CROP_REGIONS[config.feedback_step].items():
                tl = transform_to_projection_space(box["left"], box["top"], center, track_mat)
                br = transform_to_projection_space(box["left"] + box["width"], box["top"] + box["height"], center, track_mat)
                if tl and br:
                    pygame.draw.rect(surface, feedback_color, (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]))

def draw_panels(surface, mode="running"):
    left, center, right = config.get_panel_rects(surface.get_width(), surface.get_height())
    for rect in [left, center, right]:
        pygame.draw.rect(surface, config.COLOR_WHITE, rect)
        pygame.draw.rect(surface, config.COLOR_BLUE, rect, 3)

    if mode == "start":
        return center

    if mode == "scan_qr":
        draw_cartesian_plane(surface, left)
        pygame.draw.rect(surface, config.COLOR_BLUE, left, 3)
        msg = font_bold.render("[ Scanning for QR code... ]", True, config.COLOR_TEXT)
        surface.blit(msg, msg.get_rect(center=center.center))
        draw_instruction_panel(surface, right)
        return center

    if mode == "qr_confirm":
        draw_cartesian_plane(surface, left)
        pygame.draw.rect(surface, config.COLOR_BLUE, left, 3)
        draw_projection_boxes(surface, center)
        draw_instruction_panel(surface, right)
        return center

    if mode == "flip_prompt":
        draw_cartesian_plane(surface, left)
        pygame.draw.rect(surface, config.COLOR_BLUE, left, 3)
        draw_projection_boxes(surface, center)
        draw_instruction_panel(surface, right)
        return center

    draw_cartesian_plane(surface, left)
    pygame.draw.rect(surface, config.COLOR_BLUE, left, 3)

    if mode == "done":
        draw_instruction_panel(surface, right)
        return center

    draw_projection_boxes(surface, center)

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

def draw_done_button(surface, center_rect, y_offset=0):
    btn_w = max(100, min(260, int(center_rect.width * 0.30)))
    btn_h = max(28, min(70, int(btn_w * 70 / 260)))
    btn_rect = pygame.Rect(
        center_rect.x + (center_rect.width - btn_w) // 2,
        center_rect.centery + y_offset - btn_h // 2,
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

def draw_new_problem_button(surface, center_rect, y_offset=0):
    btn_w = max(100, min(260, int(center_rect.width * 0.30)))
    btn_h = max(54, min(70, int(btn_w * 70 / 260)))
    btn_rect = pygame.Rect(
        center_rect.x + (center_rect.width - btn_w) // 2,
        center_rect.centery + y_offset - btn_h // 2,
        btn_w,
        btn_h
    )
    mx, my = pygame.mouse.get_pos()
    hovered = btn_rect.collidepoint(mx, my)
    small_font = pygame.font.SysFont("segoeui", 20, bold=True)
    color = config.COLOR_BLUE if hovered else config.COLOR_WHITE
    line1 = small_font.render("NEW", True, color)
    line2 = small_font.render("PROBLEM", True, color)
    if hovered:
        pygame.draw.rect(surface, config.COLOR_WHITE, btn_rect, border_radius=12)
        pygame.draw.rect(surface, config.COLOR_BLUE, btn_rect, 3, border_radius=12)
    else:
        pygame.draw.rect(surface, config.COLOR_BLUE, btn_rect, border_radius=12)
    gap = 2
    total_h = line1.get_height() + gap + line2.get_height()
    start_y = btn_rect.centery - total_h // 2
    surface.blit(line1, line1.get_rect(centerx=btn_rect.centerx, top=start_y))
    surface.blit(line2, line2.get_rect(centerx=btn_rect.centerx, top=start_y + line1.get_height() + gap))
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

def draw_transformed_triangle(surface, area, ox, oy, scale):
    """Draws a parallelogram from the basis vectors that morphs into the transformed shape."""
    vectors = {v["label"]: v for v in config.active_vectors}

    if not all(k in vectors for k in ("u", "w", "L(u)", "L(w)")):
        return

    u = vectors["u"]
    w = vectors["w"]
    lu = vectors["L(u)"]
    lw = vectors["L(w)"]

    # Compute transformation matrix M = [L(u)|L(w)] x [u|w]^(-1)
    det = u["x"] * w["y"] - u["y"] * w["x"]
    if abs(det) < 1e-10:
        return

    m00 = (lu["x"] * w["y"] - lw["x"] * u["y"]) / det
    m01 = (-lu["x"] * w["x"] + lw["x"] * u["x"]) / det
    m10 = (lu["y"] * w["y"] - lw["y"] * u["y"]) / det
    m11 = (-lu["y"] * w["x"] + lw["y"] * u["x"]) / det

    config.matrix_engine.set_target(m00, m01, m10, m11)

    # Parallelogram formed by basis vectors: origin, u, u+w, w
    base_vertices = [
        [0.0, 0.0],
        [float(u["x"]), float(u["y"])],
        [float(u["x"] + w["x"]), float(u["y"] + w["y"])],
        [float(w["x"]), float(w["y"])]
    ]

    current_progress = getattr(config, "transformation_progress", 0.0)
    vertices = config.matrix_engine.transform_shape(base_vertices, current_progress)

    r = int(59 + (34 - 59) * current_progress)
    g = int(130 + (197 - 130) * current_progress)
    b = int(246 + (94 - 246) * current_progress)
    color = (r, g, b)

    pixel_points = []
    for x, y in vertices:
        px = ox + int(x * scale)
        py = oy - int(y * scale)
        pixel_points.append((px, py))

    clip_rect = surface.get_clip()
    surface.set_clip(area)
    poly_surf = pygame.Surface((area.width, area.height), pygame.SRCALPHA)
    local_pixels = [(pt[0] - area.x, pt[1] - area.y) for pt in pixel_points]
    pygame.draw.polygon(poly_surf, (*color, 65), local_pixels)
    surface.blit(poly_surf, (area.x, area.y))
    pygame.draw.polygon(surface, color, pixel_points, 3)
    surface.set_clip(clip_rect)


def draw_pen_cursor(surface, pos):
    cx, cy = pos
    r = 12
    pygame.draw.circle(surface, (30, 30, 30), (cx, cy), r)
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), r, 2)
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 2)