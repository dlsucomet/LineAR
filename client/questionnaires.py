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

NASA_STEPS = 20
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
BTN_DISABLED_BG = (180, 185, 190)
BTN_DISABLED_TEXT = (130, 135, 140)


def draw_nasa_tlx(surface):
    sw, sh = surface.get_size()
    pad = 40
    top_y = config.TOP_BAR_HEIGHT + pad
    content_w = sw - pad * 2

    page = config.nasa_tlx_current_page
    item = NASA_TLX_ITEMS[page]

    page_label = ui.font_body.render(f"{page + 1} of {len(NASA_TLX_ITEMS)}", True, LABEL_COLOR)
    surface.blit(page_label, (sw - pad - page_label.get_width(), top_y))

    name_surf = ui.font_large.render(item["name"], True, TEXT_COLOR)
    surface.blit(name_surf, (pad, top_y))
    cy = top_y + 50

    desc_max_w = content_w
    desc_lines = ui.wrap_text(item["desc"], ui.font_body, desc_max_w)
    for li, line in enumerate(desc_lines):
        ds = ui.font_body.render(line, True, DESC_COLOR)
        surface.blit(ds, (pad, cy + li * 22))
    cy += len(desc_lines) * 22 + 50

    slider_x = pad
    slider_w = content_w
    track_y = cy
    track_h = 6

    left_lbl = ui.font_body.render(item["left"], True, LABEL_COLOR)
    right_lbl = ui.font_body.render(item["right"], True, LABEL_COLOR)
    surface.blit(left_lbl, (slider_x, track_y + track_h + 10))
    surface.blit(right_lbl, (slider_x + slider_w - right_lbl.get_width(), track_y + track_h + 10))

    track_rect = pygame.Rect(slider_x, track_y, slider_w, track_h)
    pygame.draw.rect(surface, TRACK_COLOR, track_rect, border_radius=3)

    val = config.nasa_tlx_responses[page]
    if val is not None:
        frac = (val - 1) / (NASA_STEPS - 1) if NASA_STEPS > 1 else 0
        fill_w = max(0, int(slider_w * frac))
        if fill_w > 2:
            fill_rect = pygame.Rect(slider_x, track_y, fill_w, track_h)
            pygame.draw.rect(surface, TRACK_FILL, fill_rect, border_radius=3)

    step_spacing = slider_w / (NASA_STEPS - 1) if NASA_STEPS > 1 else slider_w
    dot_cy = track_y + track_h // 2
    dot_r = 6
    for s in range(NASA_STEPS):
        dx = int(slider_x + s * step_spacing)
        is_sel = (val is not None and s + 1 == val)
        if is_sel:
            pygame.draw.circle(surface, DOT_SELECTED, (dx, dot_cy), dot_r + 2)
        else:
            pygame.draw.circle(surface, DOT_FILL, (dx, dot_cy), dot_r)
            pygame.draw.circle(surface, DOT_BORDER, (dx, dot_cy), dot_r, 2)

    has_selection = val is not None
    btn_w = 120
    btn_h = 40
    btn_x = sw - pad - btn_w
    btn_y = sh - config.BOTTOM_BAR_HEIGHT - pad - btn_h
    next_btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    mouse_pos = pygame.mouse.get_pos()
    if has_selection:
        hovered = next_btn.collidepoint(mouse_pos)
        btn_color = BTN_HOVER_BG if hovered else BTN_BG
        text_color = BTN_TEXT
    else:
        btn_color = BTN_DISABLED_BG
        text_color = BTN_DISABLED_TEXT

    pygame.draw.rect(surface, btn_color, next_btn, border_radius=6)
    if page < len(NASA_TLX_ITEMS) - 1:
        btn_label = ui.font_bold.render("Next", True, text_color)
    else:
        btn_label = ui.font_bold.render("Finish", True, text_color)
    surface.blit(btn_label, btn_label.get_rect(center=next_btn.center))

    config._questionnaire_nasa_next_btn = next_btn if has_selection else None
    config._questionnaire_nasa_slider_rect = pygame.Rect(slider_x, track_y - 20, slider_w, 60)
    config._questionnaire_nasa_item_area = (top_y, cy)


def handle_nasa_tlx_click(pos):
    if config._questionnaire_nasa_next_btn and config._questionnaire_nasa_next_btn.collidepoint(pos):
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
        step = max(1, min(NASA_STEPS, int(round(frac * (NASA_STEPS - 1) + 1))))
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
    circle_r = 11
    circle_cx_start = pad + col_w + adj_circle_pad

    row_h = 56

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
            if dist < 20:
                config.ueq_s_responses[i] = s + 1
                return "updated"
    return None
