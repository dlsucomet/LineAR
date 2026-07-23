import pygame
import config
import ui

NASA_TLX_ITEMS = [
    {
        "name": "Mental Demand",
        "desc": "How much mental and perceptual activity was required (thinking, deciding, calculating, remembering, looking, searching)? Was the task easy or demanding, simple or complex?",
        "left": "Low", "right": "High"
    },
    {
        "name": "Physical Demand",
        "desc": "How much physical activity was required (pushing, pulling, turning, controlling, activating)? Was the task slow or brisk, restful or laborious?",
        "left": "Low", "right": "High"
    },
    {
        "name": "Temporal Demand",
        "desc": "How much time pressure did you feel due to the pace of the task? Was the pace slow and leisurely or rapid and frantic?",
        "left": "Low", "right": "High"
    },
    {
        "name": "Performance",
        "desc": "How successful were you in accomplishing what you were asked to do? How satisfied were you with your performance?",
        "left": "Good", "right": "Poor"
    },
    {
        "name": "Effort",
        "desc": "How hard did you have to work (mentally and physically) to accomplish your level of performance?",
        "left": "Low", "right": "High"
    },
    {
        "name": "Frustration",
        "desc": "How insecure, discouraged, irritated, stressed, and annoyed were you during the task?",
        "left": "Low", "right": "High"
    },
]

UEQ_S_ITEMS = [
    ("obstructive",     "supportive"),
    ("complicated",     "easy"),
    ("inefficient",     "efficient"),
    ("confusing",       "clear"),
    ("boring",          "exciting"),
    ("not interesting", "interesting"),
    ("conventional",    "inventive"),
    ("usual",           "leading edge"),
]

NASA_STEPS = 21
UEQ_STEPS = 7

TRACK_COLOR = (200, 206, 214)
TRACK_FILL = (26, 58, 107)
DOT_BORDER = (26, 58, 107)
DOT_FILL = (255, 255, 255)
DOT_SELECTED = (26, 58, 107)
TEXT_COLOR = (0, 0, 0)
DESC_COLOR = (80, 80, 80)
LABEL_COLOR = (100, 100, 100)
BTN_BG = (26, 58, 107)
BTN_TEXT = (255, 255, 255)
BTN_HOVER_BG = (40, 85, 155)


def draw_nasa_tlx(surface):
    sw, sh = surface.get_size()
    pad = 28
    content_w = sw - pad * 2
    avail_h = sh - config.TOP_BAR_HEIGHT - config.BOTTOM_BAR_HEIGHT - pad * 2

    title_font = ui.pygame.font.SysFont("segoeui", 28, bold=True)
    desc_font = ui.pygame.font.SysFont("segoeui", 20)
    label_font = ui.pygame.font.SysFont("segoeui", 18)
    btn_font = ui.pygame.font.SysFont("segoeui", 20, bold=True)

    page = config.nasa_tlx_current_page
    item = NASA_TLX_ITEMS[page]

    desc_lines = ui.wrap_text(item["desc"], desc_font, content_w)
    content_h = 40 + len(desc_lines) * 24 + 30 + 8 + 10 + 20
    start_y = config.TOP_BAR_HEIGHT + pad + max(0, (avail_h - content_h) // 2)

    page_label = label_font.render(f"{page + 1} of {len(NASA_TLX_ITEMS)}", True, LABEL_COLOR)
    surface.blit(page_label, (sw - pad - page_label.get_width(), start_y))

    name_surf = title_font.render(item["name"], True, TEXT_COLOR)
    surface.blit(name_surf, (pad, start_y))
    cy = start_y + 40

    for li, line in enumerate(desc_lines):
        ds = desc_font.render(line, True, DESC_COLOR)
        surface.blit(ds, (pad, cy + li * 24))
    cy += len(desc_lines) * 24 + 30

    slider_x = pad
    slider_w = content_w
    track_y = cy
    track_h = 8

    left_lbl = label_font.render(item["left"], True, LABEL_COLOR)
    right_lbl = label_font.render(item["right"], True, LABEL_COLOR)
    surface.blit(left_lbl, (slider_x, track_y + track_h + 8))
    surface.blit(right_lbl, (slider_x + slider_w - right_lbl.get_width(), track_y + track_h + 8))

    track_rect = pygame.Rect(slider_x, track_y, slider_w, track_h)
    pygame.draw.rect(surface, TRACK_COLOR, track_rect, border_radius=4)

    val = config.nasa_tlx_responses[page]
    display_val = val if val is not None else 0

    frac = display_val / (NASA_STEPS - 1) if NASA_STEPS > 1 else 0
    fill_w = max(0, int(slider_w * frac))
    if fill_w > 2:
        fill_rect = pygame.Rect(slider_x, track_y, fill_w, track_h)
        pygame.draw.rect(surface, TRACK_FILL, fill_rect, border_radius=4)

    step_spacing = slider_w / (NASA_STEPS - 1) if NASA_STEPS > 1 else slider_w
    dot_cy = track_y + track_h // 2
    dot_r = 9
    for s in range(NASA_STEPS):
        dx = int(slider_x + s * step_spacing)
        is_sel = (s == display_val)
        if is_sel:
            pygame.draw.circle(surface, DOT_SELECTED, (dx, dot_cy), dot_r + 2)
        else:
            pygame.draw.circle(surface, DOT_FILL, (dx, dot_cy), dot_r)
            pygame.draw.circle(surface, DOT_BORDER, (dx, dot_cy), dot_r, 2)
    btn_w = 160
    btn_h = 50
    btn_x = sw - pad - btn_w
    btn_y = sh - config.BOTTOM_BAR_HEIGHT - pad - btn_h
    next_btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    mouse_pos = pygame.mouse.get_pos()
    hovered = next_btn.collidepoint(mouse_pos)
    btn_color = BTN_HOVER_BG if hovered else BTN_BG
    text_color = BTN_TEXT

    pygame.draw.rect(surface, btn_color, next_btn, border_radius=6)
    if page < len(NASA_TLX_ITEMS) - 1:
        btn_label = btn_font.render("Next", True, text_color)
    else:
        btn_label = btn_font.render("Finish", True, text_color)
    surface.blit(btn_label, btn_label.get_rect(center=next_btn.center))

    config._questionnaire_nasa_next_btn = next_btn
    config._questionnaire_nasa_slider_rect = pygame.Rect(slider_x, track_y - 20, slider_w + dot_r, 60)


def handle_nasa_tlx_click(pos):
    if config._questionnaire_nasa_next_btn and config._questionnaire_nasa_next_btn.collidepoint(pos):
        if config.nasa_tlx_responses[config.nasa_tlx_current_page] is None:
            config.nasa_tlx_responses[config.nasa_tlx_current_page] = 0
        config.nasa_tlx_current_page += 1
        if config.nasa_tlx_current_page >= len(NASA_TLX_ITEMS):
            return "done"
        return "next"

    slider_rect = config._questionnaire_nasa_slider_rect
    if slider_rect and slider_rect.collidepoint(pos):
        slider_x = slider_rect.x
        slider_w = slider_rect.w
        rel_x = pos[0] - slider_x
        frac = rel_x / slider_w
        step = max(0, min(NASA_STEPS - 1, int(round(frac * (NASA_STEPS - 1)))))
        config.nasa_tlx_responses[config.nasa_tlx_current_page] = step
        return "updated"
    return None


def draw_ueq_s(surface):
    sw, sh = surface.get_size()
    pad = 40
    top_y = config.TOP_BAR_HEIGHT + pad
    content_w = sw - pad * 2

    title = ui.font_large.render("User Experience Questionnaire (UEQ-S)", True, TEXT_COLOR)
    surface.blit(title, (pad, top_y))
    cy = top_y + 50

    col_w = 160
    adj_circle_pad = 28
    circle_area_w = content_w - col_w * 2 - adj_circle_pad * 2
    circle_spacing = circle_area_w / (UEQ_STEPS - 1) if UEQ_STEPS > 1 else circle_area_w
    circle_r = 16
    circle_cx_start = pad + col_w + adj_circle_pad

    row_h = 64

    for i, (left_adj, right_adj) in enumerate(UEQ_S_ITEMS):
        row_y = cy + i * row_h
        center_y = row_y + row_h // 2

        left_surf = ui.font_bold.render(left_adj, True, TEXT_COLOR)
        surface.blit(left_surf, (pad + col_w - left_surf.get_width(), center_y - left_surf.get_height() // 2))

        right_surf = ui.font_bold.render(right_adj, True, TEXT_COLOR)
        surface.blit(right_surf, (pad + col_w + adj_circle_pad + circle_area_w + adj_circle_pad, center_y - right_surf.get_height() // 2))

        val = config.ueq_s_responses[i]
        for s in range(UEQ_STEPS):
            sx = circle_cx_start + int(s * circle_spacing)
            is_sel = (val is not None and s + 1 == val)
            if is_sel:
                pygame.draw.circle(surface, DOT_SELECTED, (sx, center_y), circle_r + 1)
            else:
                pygame.draw.circle(surface, DOT_FILL, (sx, center_y), circle_r)
                pygame.draw.circle(surface, DOT_BORDER, (sx, center_y), circle_r, 2)

    btn_w = 120
    btn_h = 40
    btn_x = sw - pad - btn_w
    btn_y = sh - config.BOTTOM_BAR_HEIGHT - pad - btn_h
    submit_btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    mouse_pos = pygame.mouse.get_pos()
    hovered = submit_btn.collidepoint(mouse_pos)
    btn_color = BTN_HOVER_BG if hovered else BTN_BG
    pygame.draw.rect(surface, btn_color, submit_btn, border_radius=6)
    btn_label = ui.font_bold.render("Submit", True, BTN_TEXT)
    surface.blit(btn_label, btn_label.get_rect(center=submit_btn.center))

    config._questionnaire_ueq_submit_btn = submit_btn
    config._questionnaire_ueq_circles = (circle_cx_start, circle_spacing, circle_area_w)
    config._questionnaire_ueq_row_h = row_h
    config._questionnaire_ueq_title_bottom = cy


def handle_ueq_s_click(pos):
    if config._questionnaire_ueq_submit_btn and config._questionnaire_ueq_submit_btn.collidepoint(pos):
        return "submit"

    circle_cx_start, circle_spacing, _ = config._questionnaire_ueq_circles
    title_bottom = config._questionnaire_ueq_title_bottom
    row_h = config._questionnaire_ueq_row_h

    for i in range(len(UEQ_S_ITEMS)):
        row_y = title_bottom + i * row_h
        center_y = row_y + row_h // 2
        for s in range(UEQ_STEPS):
            sx = circle_cx_start + int(s * circle_spacing)
            dist = ((pos[0] - sx) ** 2 + (pos[1] - center_y) ** 2) ** 0.5
            if dist < 25:
                config.ueq_s_responses[i] = s + 1
                return "updated"
    return None
