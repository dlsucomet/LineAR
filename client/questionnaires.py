import pygame
import config
from ui import font_large, font_medium, font_body, font_bold

NASA_TLX_ITEMS = [
    {"name": "Mental Demand",       "left": "Low",  "right": "High"},
    {"name": "Physical Demand",     "left": "Low",  "right": "High"},
    {"name": "Temporal Demand",     "left": "Low",  "right": "High"},
    {"name": "Performance",         "left": "Good", "right": "Poor"},
    {"name": "Effort",              "left": "Low",  "right": "High"},
    {"name": "Frustration",         "left": "Low",  "right": "High"},
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
TRACK_COLOR = (210, 215, 222)
ACTIVE_COLOR = (26, 58, 107)
TEXT_COLOR = (0, 0, 0)
LABEL_COLOR = (60, 60, 60)
BTN_BG = (26, 58, 107)
BTN_TEXT = (255, 255, 255)
BTN_HOVER_BG = (40, 85, 155)

def draw_nasa_tlx(surface):
    sw, sh = surface.get_size()
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    pad = 28
    row_h = 56
    title_h = 44
    subtitle_h = 28
    btn_h = 40
    inner_btn_pad = 20
    rows_count = len(NASA_TLX_ITEMS)

    card_content_h = title_h + subtitle_h + rows_count * row_h + btn_h + pad * 3
    card_h = min(card_content_h, sh - 60)
    card_w = min(820, sw - 80)
    card_x = (sw - card_w) // 2
    card_y = (sh - card_h) // 2

    pygame.draw.rect(surface, CARD_BG, (card_x, card_y, card_w, card_h), border_radius=10)
    pygame.draw.rect(surface, CARD_BORDER, (card_x, card_y, card_w, card_h), 3, border_radius=10)

    cx = card_x + pad
    cy = card_y + pad

    title = font_large.render("NASA Task Load Index (NASA-TLX)", True, TEXT_COLOR)
    surface.blit(title, (cx, cy))
    cy += title_h

    subtitle = font_body.render("Click on each scale to rate the task", True, LABEL_COLOR)
    surface.blit(subtitle, (cx, cy))
    cy += subtitle_h

    label_col_w = 180
    value_col_w = 50
    slider_x = cx + label_col_w
    slider_w = card_w - pad * 2 - label_col_w - value_col_w - 12

    btn_w = 120
    btn_x = card_x + card_w - pad - btn_w
    btn_y = card_y + card_h - pad - btn_h
    next_btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    for i, item in enumerate(NASA_TLX_ITEMS):
        row_y = cy + i * row_h

        label = font_bold.render(item["name"], True, TEXT_COLOR)
        surface.blit(label, (cx, row_y + (row_h - label.get_height()) // 2))

        left_lbl = font_body.render(item["left"], True, LABEL_COLOR)
        right_lbl = font_body.render(item["right"], True, LABEL_COLOR)

        left_lbl_x = slider_x
        right_lbl_x = slider_x + slider_w - right_lbl.get_width()

        label_bottom_y = row_y + row_h

        surface.blit(left_lbl, (left_lbl_x, label_bottom_y - left_lbl.get_height() - 2))
        surface.blit(right_lbl, (right_lbl_x, label_bottom_y - right_lbl.get_height() - 2))

        track_y = row_y + row_h // 2 - 4
        track_h = 8
        track_rect = pygame.Rect(slider_x, track_y, slider_w, track_h)

        pygame.draw.rect(surface, TRACK_COLOR, track_rect, border_radius=4)

        val = config.nasa_tlx_responses[i]
        if val is not None:
            frac = (val - 1) / (NASA_STEPS - 1) if NASA_STEPS > 1 else 0
            fill_w = int(slider_w * frac)
            if fill_w > 0:
                fill_rect = pygame.Rect(slider_x, track_y, fill_w, track_h)
                pygame.draw.rect(surface, ACTIVE_COLOR, fill_rect, border_radius=4)

        step_spacing = slider_w / (NASA_STEPS - 1) if NASA_STEPS > 1 else slider_w
        for s in range(NASA_STEPS):
            sx = slider_x + int(s * step_spacing)
            radius = 4 if (val is not None and s + 1 == val) else 3
            color = ACTIVE_COLOR if (val is not None and s + 1 == val) else TRACK_COLOR
            pygame.draw.circle(surface, color, (sx, track_y + track_h // 2), radius)

        if val is not None:
            val_text = font_bold.render(str(val), True, ACTIVE_COLOR)
            surface.blit(val_text, (slider_x + slider_w + 8, row_y + (row_h - val_text.get_height()) // 2))

    mouse_pos = pygame.mouse.get_pos()
    hovered = next_btn.collidepoint(mouse_pos)
    btn_color = BTN_HOVER_BG if hovered else BTN_BG
    pygame.draw.rect(surface, btn_color, next_btn, border_radius=6)
    btn_label = font_bold.render("Next", True, BTN_TEXT)
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
        row_y = subtitle_bottom + i * row_h
        slider_rect = pygame.Rect(slider_x, row_y, slider_w, row_h)
        if slider_rect.collidepoint(pos):
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
    row_h = 50
    title_h = 44
    btn_h = 40
    rows_count = len(UEQ_S_ITEMS)

    card_content_h = title_h + rows_count * row_h + btn_h + pad * 3
    card_h = min(card_content_h, sh - 60)
    card_w = min(820, sw - 80)
    card_x = (sw - card_w) // 2
    card_y = (sh - card_h) // 2

    pygame.draw.rect(surface, CARD_BG, (card_x, card_y, card_w, card_h), border_radius=10)
    pygame.draw.rect(surface, CARD_BORDER, (card_x, card_y, card_w, card_h), 3, border_radius=10)

    cx = card_x + pad
    cy = card_y + pad

    title = font_large.render("User Experience Questionnaire (UEQ-S)", True, TEXT_COLOR)
    surface.blit(title, (cx, cy))
    cy += title_h

    col_w = 150
    circle_area_w = card_w - pad * 2 - col_w * 2
    circle_spacing = circle_area_w / (UEQ_STEPS - 1) if UEQ_STEPS > 1 else circle_area_w
    circle_radius = 10
    circle_center_x_start = cx + col_w

    btn_w = 120
    btn_x = card_x + card_w - pad - btn_w
    btn_y = card_y + card_h - pad - btn_h
    submit_btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    for i, (left_adj, right_adj) in enumerate(UEQ_S_ITEMS):
        row_y = cy + i * row_h
        center_y = row_y + row_h // 2

        left_surf = font_bold.render(left_adj, True, TEXT_COLOR)
        surface.blit(left_surf, (cx + col_w - left_surf.get_width() - 8, center_y - left_surf.get_height() // 2))

        right_surf = font_bold.render(right_adj, True, TEXT_COLOR)
        surface.blit(right_surf, (cx + col_w + circle_area_w + 8, center_y - right_surf.get_height() // 2))

        val = config.ueq_s_responses[i]
        for s in range(UEQ_STEPS):
            sx = circle_center_x_start + int(s * circle_spacing)
            selected = (val is not None and s + 1 == val)
            color = ACTIVE_COLOR if selected else TRACK_COLOR
            r = circle_radius if selected else circle_radius - 2
            pygame.draw.circle(surface, color, (sx, center_y), r)
            if not selected:
                pygame.draw.circle(surface, ACTIVE_COLOR, (sx, center_y), r, 2)

    mouse_pos = pygame.mouse.get_pos()
    hovered = submit_btn.collidepoint(mouse_pos)
    btn_color = BTN_HOVER_BG if hovered else BTN_BG
    pygame.draw.rect(surface, btn_color, submit_btn, border_radius=6)
    btn_label = font_bold.render("Submit", True, BTN_TEXT)
    surface.blit(btn_label, btn_label.get_rect(center=submit_btn.center))

    config._questionnaire_ueq_submit_btn = submit_btn
    config._questionnaire_ueq_card = (card_x, card_y, card_w, card_h)
    config._questionnaire_ueq_circles = (circle_center_x_start, circle_spacing, circle_area_w)
    config._questionnaire_ueq_row_h = row_h
    config._questionnaire_ueq_title_bottom = cy


def handle_ueq_s_click(pos):
    if config._questionnaire_ueq_submit_btn and config._questionnaire_ueq_submit_btn.collidepoint(pos):
        return "submit"

    card_x, card_y, card_w, card_h = config._questionnaire_ueq_card
    circle_center_x_start, circle_spacing, _ = config._questionnaire_ueq_circles
    title_bottom = config._questionnaire_ueq_title_bottom
    row_h = config._questionnaire_ueq_row_h
    col_w = 150
    circle_area_w = card_w - 28 * 2 - col_w * 2

    for i in range(len(UEQ_S_ITEMS)):
        row_y = title_bottom + i * row_h
        for s in range(UEQ_STEPS):
            sx = circle_center_x_start + int(s * circle_spacing)
            dist = ((pos[0] - sx) ** 2 + (pos[1] - (row_y + row_h // 2)) ** 2) ** 0.5
            if dist < 18:
                config.ueq_s_responses[i] = s + 1
                return "updated"
    return None
