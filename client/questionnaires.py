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

CARD_BG = (248, 250, 252)
CARD_BORDER = (26, 58, 107)
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
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    pad = 28
    name_h = 18
    desc_h = 34
    slider_h = 28
    row_h = name_h + desc_h + slider_h + 6
    title_h = 40
    btn_h = 40
    rows_count = len(NASA_TLX_ITEMS)

    card_content_h = title_h + rows_count * row_h + btn_h + pad * 3
    card_h = min(card_content_h, sh - 60)
    card_w = min(860, sw - 80)
    card_x = (sw - card_w) // 2
    card_y = (sh - card_h) // 2

    pygame.draw.rect(surface, CARD_BG, (card_x, card_y, card_w, card_h), border_radius=10)
    pygame.draw.rect(surface, CARD_BORDER, (card_x, card_y, card_w, card_h), 3, border_radius=10)

    cx = card_x + pad
    cy = card_y + pad

    title = ui.font_large.render("NASA Task Load Index (NASA-TLX)", True, TEXT_COLOR)
    surface.blit(title, (cx, cy))
    cy += title_h

    slider_x = cx
    slider_w = card_w - pad * 2
    desc_max_w = slider_w - 180

    btn_w = 120
    btn_x = card_x + card_w - pad - btn_w
    btn_y = card_y + card_h - pad - btn_h
    next_btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    for i, item in enumerate(NASA_TLX_ITEMS):
        ry = cy + i * row_h

        name_surf = ui.font_bold.render(item["name"], True, TEXT_COLOR)
        surface.blit(name_surf, (cx, ry))

        desc_lines = ui.wrap_text(item["desc"], ui.font_body, desc_max_w) if "desc" in item else []
        for li, line in enumerate(desc_lines[:2]):
            ds = ui.font_body.render(line, True, DESC_COLOR)
            surface.blit(ds, (cx + 2, ry + name_h + 2 + li * 16))

        track_y = ry + name_h + desc_h + 2
        track_h = 6

        left_lbl = ui.font_body.render(item["left"], True, LABEL_COLOR)
        right_lbl = ui.font_body.render(item["right"], True, LABEL_COLOR)
        surface.blit(left_lbl, (slider_x, track_y + track_h + 4))
        surface.blit(right_lbl, (slider_x + slider_w - right_lbl.get_width(), track_y + track_h + 4))

        track_rect = pygame.Rect(slider_x, track_y, slider_w, track_h)
        pygame.draw.rect(surface, TRACK_COLOR, track_rect, border_radius=3)

        val = config.nasa_tlx_responses[i]
        if val is not None:
            frac = (val - 1) / (NASA_STEPS - 1) if NASA_STEPS > 1 else 0
            fill_w = max(0, int(slider_w * frac))
            if fill_w > 2:
                fill_rect = pygame.Rect(slider_x, track_y, fill_w, track_h)
                pygame.draw.rect(surface, TRACK_FILL, fill_rect, border_radius=3)

        step_spacing = slider_w / (NASA_STEPS - 1) if NASA_STEPS > 1 else slider_w
        dot_cy = track_y + track_h // 2
        dot_r = 5
        for s in range(NASA_STEPS):
            dx = int(slider_x + s * step_spacing)
            is_sel = (val is not None and s + 1 == val)
            if is_sel:
                pygame.draw.circle(surface, DOT_SELECTED, (dx, dot_cy), dot_r + 1)
            else:
                pygame.draw.circle(surface, DOT_FILL, (dx, dot_cy), dot_r)
                pygame.draw.circle(surface, DOT_BORDER, (dx, dot_cy), dot_r, 2)

    mouse_pos = pygame.mouse.get_pos()
    hovered = next_btn.collidepoint(mouse_pos)
    btn_color = BTN_HOVER_BG if hovered else BTN_BG
    pygame.draw.rect(surface, btn_color, next_btn, border_radius=6)
    btn_label = ui.font_bold.render("Next", True, BTN_TEXT)
    surface.blit(btn_label, btn_label.get_rect(center=next_btn.center))

    config._questionnaire_nasa_next_btn = next_btn
    config._questionnaire_nasa_card = (card_x, card_y, card_w, card_h)
    config._questionnaire_nasa_slider = (slider_x, slider_w)
    config._questionnaire_nasa_row_h = row_h
    config._questionnaire_nasa_subtitle_bottom = cy


def handle_nasa_tlx_click(pos):
    if config._questionnaire_nasa_next_btn and config._questionnaire_nasa_next_btn.collidepoint(pos):
        return "next"

    card_x, card_y, card_w, card_h = config._questionnaire_nasa_card
    slider_x, slider_w = config._questionnaire_nasa_slider
    subtitle_bottom = config._questionnaire_nasa_subtitle_bottom
    row_h = config._questionnaire_nasa_row_h

    for i in range(len(NASA_TLX_ITEMS)):
        ry = subtitle_bottom + i * row_h
        click_rect = pygame.Rect(slider_x, ry, slider_w, row_h)
        if click_rect.collidepoint(pos):
            rel_x = pos[0] - slider_x
            frac = rel_x / slider_w
            step = max(1, min(NASA_STEPS, int(round(frac * (NASA_STEPS - 1) + 1))))
            config.nasa_tlx_responses[i] = step
            return "updated"
    return None


def draw_ueq_s(surface):
    sw, sh = surface.get_size()
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    pad = 28
    row_h = 56
    title_h = 40
    btn_h = 40
    rows_count = len(UEQ_S_ITEMS)

    card_content_h = title_h + rows_count * row_h + btn_h + pad * 3
    card_h = min(card_content_h, sh - 60)
    card_w = min(860, sw - 80)
    card_x = (sw - card_w) // 2
    card_y = (sh - card_h) // 2

    pygame.draw.rect(surface, CARD_BG, (card_x, card_y, card_w, card_h), border_radius=10)
    pygame.draw.rect(surface, CARD_BORDER, (card_x, card_y, card_w, card_h), 3, border_radius=10)

    cx = card_x + pad
    cy = card_y + pad

    title = ui.font_large.render("User Experience Questionnaire (UEQ-S)", True, TEXT_COLOR)
    surface.blit(title, (cx, cy))
    cy += title_h

    col_w = 160
    adj_circle_pad = 28
    circle_area_w = card_w - pad * 2 - col_w * 2 - adj_circle_pad * 2
    circle_spacing = circle_area_w / (UEQ_STEPS - 1) if UEQ_STEPS > 1 else circle_area_w
    circle_r = 11
    circle_cx_start = cx + col_w + adj_circle_pad

    btn_w = 120
    btn_x = card_x + card_w - pad - btn_w
    btn_y = card_y + card_h - pad - btn_h
    submit_btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    for i, (left_adj, right_adj) in enumerate(UEQ_S_ITEMS):
        row_y = cy + i * row_h
        center_y = row_y + row_h // 2

        left_surf = ui.font_bold.render(left_adj, True, TEXT_COLOR)
        surface.blit(left_surf, (cx + col_w - left_surf.get_width(), center_y - left_surf.get_height() // 2))

        right_surf = ui.font_bold.render(right_adj, True, TEXT_COLOR)
        surface.blit(right_surf, (cx + col_w + adj_circle_pad + circle_area_w + adj_circle_pad, center_y - right_surf.get_height() // 2))

        val = config.ueq_s_responses[i]
        for s in range(UEQ_STEPS):
            sx = circle_cx_start + int(s * circle_spacing)
            is_sel = (val is not None and s + 1 == val)
            if is_sel:
                pygame.draw.circle(surface, DOT_SELECTED, (sx, center_y), circle_r + 1)
            else:
                pygame.draw.circle(surface, DOT_FILL, (sx, center_y), circle_r)
                pygame.draw.circle(surface, DOT_BORDER, (sx, center_y), circle_r, 2)

    mouse_pos = pygame.mouse.get_pos()
    hovered = submit_btn.collidepoint(mouse_pos)
    btn_color = BTN_HOVER_BG if hovered else BTN_BG
    pygame.draw.rect(surface, btn_color, submit_btn, border_radius=6)
    btn_label = ui.font_bold.render("Submit", True, BTN_TEXT)
    surface.blit(btn_label, btn_label.get_rect(center=submit_btn.center))

    config._questionnaire_ueq_submit_btn = submit_btn
    config._questionnaire_ueq_card = (card_x, card_y, card_w, card_h)
    config._questionnaire_ueq_circles = (circle_cx_start, circle_spacing, circle_area_w)
    config._questionnaire_ueq_row_h = row_h
    config._questionnaire_ueq_title_bottom = cy


def handle_ueq_s_click(pos):
    if config._questionnaire_ueq_submit_btn and config._questionnaire_ueq_submit_btn.collidepoint(pos):
        return "submit"

    card_x, card_y, card_w, card_h = config._questionnaire_ueq_card
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
