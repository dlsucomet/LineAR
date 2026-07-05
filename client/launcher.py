import os
import sys
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", message="pkg_resources")

import pygame
import runner

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BG = (248, 250, 252)
COLOR_BLUE = (26, 58, 107)

BTN_WIDTH = 220
BTN_HEIGHT = 50
BTN_GAP = 20

def main():
    while True:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("LineAR - Launcher")
        clock = pygame.time.Clock()

        font_large = pygame.font.SysFont("segoeui", 42, bold=True)
        font_btn = pygame.font.SysFont("segoeui", 20, bold=True)

        center_x = WINDOW_WIDTH // 2
        btn_y = 230

        highlights_rect = pygame.Rect(0, btn_y, BTN_WIDTH, BTN_HEIGHT)
        highlights_rect.centerx = center_x - BTN_WIDTH // 2 - BTN_GAP // 2

        no_highlights_rect = pygame.Rect(0, btn_y, BTN_WIDTH, BTN_HEIGHT)
        no_highlights_rect.centerx = center_x + BTN_WIDTH // 2 + BTN_GAP // 2

        running = True
        while running:
            mx, my = pygame.mouse.get_pos()

            hl_hover = highlights_rect.collidepoint(mx, my)
            no_hover = no_highlights_rect.collidepoint(mx, my)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if highlights_rect.collidepoint(event.pos):
                        pygame.quit()
                        runner.main("highlights")
                        running = False
                        break
                    elif no_highlights_rect.collidepoint(event.pos):
                        pygame.quit()
                        runner.main("no_highlights")
                        running = False
                        break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            if not running:
                break

            screen.fill(COLOR_BG)

            title = font_large.render("LineAR", True, COLOR_BLACK)
            title_rect = title.get_rect(center=(center_x, 100))
            screen.blit(title, title_rect)

            if hl_hover:
                pygame.draw.rect(screen, COLOR_WHITE, highlights_rect, border_radius=6)
                pygame.draw.rect(screen, COLOR_BLUE, highlights_rect, 3, border_radius=6)
                hl_text = font_btn.render("With Highlights", True, COLOR_BLUE)
            else:
                pygame.draw.rect(screen, COLOR_BLUE, highlights_rect, border_radius=6)
                hl_text = font_btn.render("With Highlights", True, COLOR_WHITE)
            hl_text_rect = hl_text.get_rect(center=highlights_rect.center)
            screen.blit(hl_text, hl_text_rect)

            if no_hover:
                pygame.draw.rect(screen, COLOR_WHITE, no_highlights_rect, border_radius=6)
                pygame.draw.rect(screen, COLOR_BLUE, no_highlights_rect, 3, border_radius=6)
                no_text = font_btn.render("No Highlights", True, COLOR_BLUE)
            else:
                pygame.draw.rect(screen, COLOR_BLUE, no_highlights_rect, border_radius=6)
                no_text = font_btn.render("No Highlights", True, COLOR_WHITE)
            no_text_rect = no_text.get_rect(center=no_highlights_rect.center)
            screen.blit(no_text, no_text_rect)

            pygame.display.flip()
            clock.tick(30)


if __name__ == "__main__":
    main()
