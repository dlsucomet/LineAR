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
        "math": "c1 * [transformed 1] + c2 * [transformed 2]"
    },
    "secondStepLeft": {
        "title": "Scalar Multiplication (Left Component)",
        "desc": "Distribute the first scalar coefficient into the left vector component elements.",
        "math": "c * [x, y]^T = [c*x, c*y]^T"
    },
    "secondStepRight": {
        "title": "Scalar Multiplication (Right Component)",
        "desc": "Distribute the second scalar coefficient into the right vector component elements.",
        "math": "c * [x, y]^T = [c*x, c*y]^T"
    },
    "thirdStep": {
        "title": "Vector Addition (Final Matrix)",
        "desc": "Perform row-by-row matrix addition on your scaled vector elements to solve.",
        "math": "[a1, b1]^T + [a2, b2]^T = [a1+a2, b1+b2]^T"
    },
    "complete": {
        "title": "Problem Completed!",
        "desc": "The linear transformation mapping operations match the coordinate target state vector space outputs.",
        "math": "L(v) = [14, -7]^T"
    }
}

def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        test = current + (" " if current else "") + word
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current: lines.append(current)
            current = word
    if current: lines.append(current)
    return lines

def draw_top_bar(surface):
    bar_rect = pygame.Rect(0, 0, surface.get_width(), config.TOP_BAR_HEIGHT)
    pygame.draw.rect(surface, config.COLOR_BLUE, bar_rect)
    text = font_large.render("Place paper on the designated projection area", True, config.COLOR_WHITE)
    surface.blit(text, text.get_rect(center=(surface.get_width() // 2, config.TOP_BAR_HEIGHT // 2)))

def draw_bottom_bar(surface):
    H = surface.get_height()
    pygame.draw.rect(surface, config.COLOR_BG, pygame.Rect(0, H - config.BOTTOM_BAR_HEIGHT, surface.get_width(), config.BOTTOM_BAR_HEIGHT))
    console = font_bold.render(f"CONSOLE LOG: {config.status_msg}", True, config.COLOR_TEXT)
    surface.blit(console, (config.OUTER_GAP, H - config.BOTTOM_BAR_HEIGHT + (config.BOTTOM_BAR_HEIGHT - console.get_height()) // 2))

def draw_cartesian_plane(surface, area):
    pygame.draw.rect(surface, config.COLOR_WHITE, area)
    ox, oy = area.x + area.width // 2, area.y + area.height // 2
    scale = max(1, int(min(area.width, area.height) / 14))
    grid_surf = pygame.Surface((area.width, area.height), pygame.SRCALPHA)
    for x in range(ox % scale, area.x + area.width, scale):
        pygame.draw.line(grid_surf, (*config.COLOR_GRID, 180), (x, area.y), (x, area.y + area.height))
    for y in range(oy % scale, area.y + area.height, scale):
        pygame.draw.line(grid_surf, (*config.COLOR_GRID, 180), (area.x, y), (area.x + area.width, y))
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
    
    surface.blit(font_large.render(step_info["title"], True, config.COLOR_TEXT), (cx, cy))
    cy += 40
    for line in wrap_text(step_info["desc"], font_body, mw):
        surface.blit(font_body.render(line, True, config.COLOR_TEXT), (cx, cy))
        cy += 22
    cy += 10
    eq_rect = pygame.Rect(cx, cy, mw, 50)
    pygame.draw.rect(surface, config.COLOR_WHITE, eq_rect)
    pygame.draw.rect(surface, config.COLOR_TEXT, eq_rect, 2)
    eq_t = font_equation.render(step_info["math"], True, config.COLOR_TEXT)
    surface.blit(eq_t, eq_t.get_rect(center=eq_rect.center))

def draw_panels(surface, mode="running"):
    left, center, right = config.get_panel_rects(surface.get_width(), surface.get_height())
    for rect in [left, center, right]:
        pygame.draw.rect(surface, config.COLOR_WHITE, rect)
        pygame.draw.rect(surface, config.COLOR_BLUE, rect, 3)
        
    if mode == "start":
        return center
        
    draw_cartesian_plane(surface, left)
    pygame.draw.rect(surface, config.COLOR_BLUE, left, 3)
    
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
            
    draw_instruction_panel(surface, right)
    return center

def draw_start_button(surface, center_rect):
    btn_w, btn_h = 260, 70
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
    rect = pygame.Rect(center_rect.centerx - 80, surface.get_height() - config.BOTTOM_BAR_HEIGHT + (config.BOTTOM_BAR_HEIGHT - 44) // 2, 160, 44)
    hovered = rect.collidepoint(pygame.mouse.get_pos())
    pygame.draw.rect(surface, config.COLOR_WHITE if hovered else config.COLOR_BLUE, rect, border_radius=6)
    pygame.draw.rect(surface, config.COLOR_BLUE, rect, 3, border_radius=6)
    t = font_medium.render("End Task", True, config.COLOR_BLUE if hovered else config.COLOR_WHITE)
    surface.blit(t, t.get_rect(center=rect.center))
    return rect