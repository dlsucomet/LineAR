# Disable Console Log

## File: `client/ui.py` — `draw_bottom_bar()` function

Comment out lines 156–179 (the entire `if msg:` block):

```python
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
            show_hints = getattr(config, "debug_mode", False) and getattr(
                config, "debug_hints", False)
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
```

## Why it's safe

- `config.status_msg` is write-only from `log_message()` — no logic depends on it
- `wrap_right` and `avail_w` are local to this block
- Debug hints (line 181) and debug button (line 190) are independent
- Bottom bar background still renders (line 153)
