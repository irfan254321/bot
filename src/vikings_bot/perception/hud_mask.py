"""Static RYL HUD exclusion masking and self-character zone computation (1920x1080 & 1024x768 VM)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def get_static_hud_rects(
    frame_w: int,
    frame_h: int,
    *,
    hud_fixed_pixel_ui: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Return static RYL HUD rectangles (x1, y1, x2, y2) masked before OpenCV detection.

    Covers:
    1. Top-left minimap + coordinate bar
    2. Upper-left party status window
    3. Left vertical skill hotbar
    4. Bottom-left HP/MP/EA status panel
    5. Bottom-right chat log window
    """
    fw = max(1, int(frame_w))
    fh = max(1, int(frame_h))
    if hud_fixed_pixel_ui:
        # RYL renders HUD elements at fixed 1:1 pixel sizes at 1024x768 VMware
        return [
            (0, 0, min(fw, 132), min(fh, 152)),  # minimap + coords
            (0, 140, min(fw, 200), min(fh, 255)),  # party panel
            (0, max(0, fh - 415), min(fw, 52), fh),  # left vertical skill bar
            (0, max(0, fh - 96), min(fw, 265), fh),  # bottom-left HP/MP/EA panel
            (max(0, fw - 332), max(0, fh - 245), fw, fh),  # bottom-right chat window
        ]
    return [
        (0, 0, int(round(fw * 0.070)), int(round(fh * 0.140))),  # minimap + coords
        (0, int(round(fh * 0.130)), int(round(fw * 0.115)), int(round(fh * 0.245))),  # party panel
        (0, int(round(fh * 0.600)), int(round(fw * 0.032)), fh),  # left skill bar
        (0, int(round(fh * 0.905)), int(round(fw * 0.145)), fh),  # bottom-left HP/MP/EA panel
        (int(round(fw * 0.695)), int(round(fh * 0.765)), fw, fh),  # bottom-right chat window
    ]


@lru_cache(maxsize=16)
def build_playfield_hud_mask(
    frame_w: int,
    frame_h: int,
    playfield_rect: tuple[int, int, int, int],
    hud_fixed_pixel_ui: bool = False,
) -> np.ndarray:
    """Build cached uint8 mask (255 = allowed playfield, 0 = static HUD) aligned to playfield_rect."""
    px1, py1, px2, py2 = playfield_rect
    pw = max(1, int(px2 - px1))
    ph = max(1, int(py2 - py1))
    mask = np.full((ph, pw), 255, dtype=np.uint8)
    for hx1, hy1, hx2, hy2 in get_static_hud_rects(frame_w, frame_h, hud_fixed_pixel_ui=hud_fixed_pixel_ui):
        lx1 = max(0, hx1 - px1)
        ly1 = max(0, hy1 - py1)
        lx2 = min(pw, hx2 - px1)
        ly2 = min(ph, hy2 - py1)
        if lx2 > lx1 and ly2 > ly1:
            mask[ly1:ly2, lx1:lx2] = 0
    return mask


def compute_self_zone(
    frame_w: int,
    frame_h: int,
    player_center_norm: tuple[float, float] = (0.50, 0.52),
    hud_fixed_pixel_ui: bool = False,
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Return ((player_feet_x, player_feet_y), self_nameplate_exclusion_box) in full frame coords."""
    fw = max(1, int(frame_w))
    fh = max(1, int(frame_h))
    sx = fw / 1920.0
    sy = fh / 1080.0
    px = int(round(fw * player_center_norm[0]))
    py = int(round(fh * player_center_norm[1]))
    feet_y = int(round(fh * (player_center_norm[1] + 0.032)))

    half_w = int(round(max(62.0, 92.0 * sx) if hud_fixed_pixel_ui else max(56.0, 88.0 * sx)))
    top_h = int(round(max(78.0, 118.0 * sy) if hud_fixed_pixel_ui else max(72.0, 118.0 * sy)))
    bot_h = int(round(max(36.0, 48.0 * sy)))
    self_box = (
        max(0, px - half_w),
        max(0, py - top_h),
        min(fw, px + half_w),
        min(fh, py + bot_h),
    )
    return ((px, feet_y), self_box)
