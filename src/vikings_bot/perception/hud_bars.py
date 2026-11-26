"""Fast column-wise HP and MP bar reader resilient to centered text overlays."""

from __future__ import annotations

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig


class HudBarReader:
    """Reads normalized HP (red) and MP (blue/purple) fill ratios from bottom-left HUD."""

    def __init__(self, config: BotConfig) -> None:
        self.hp_roi = config.roi("hp_bar")
        self.mp_roi = config.roi("mp_bar")

    @staticmethod
    def _measure_bar_fill(mask: np.ndarray) -> float:
        if mask.size == 0 or mask.shape[1] == 0:
            return 0.0
        # A column is considered filled if at least one pixel in that column has bar color
        # (top/bottom edges of bar remain visible even behind white numeric overlay text).
        col_filled = mask.any(axis=0).astype(np.uint8)[None, :]
        # Bridge narrow 1-2px gaps where white text strokes span full height
        kernel = np.ones((1, 5), dtype=np.uint8)
        closed = cv2.morphologyEx(col_filled, cv2.MORPH_CLOSE, kernel)
        filled_cols = np.where(closed[0] > 0)[0]
        if len(filled_cols) == 0:
            return 0.0
        # Bar fills left-to-right; use blend of rightmost filled pixel and total coverage
        # to reject stray single-pixel noise on the far right while keeping text resilience.
        coverage = float(closed[0].mean())
        rightmost = float(filled_cols[-1] + 1) / float(mask.shape[1])
        if rightmost - coverage > 0.20:
            return max(0.0, min(1.0, coverage))
        return max(0.0, min(1.0, max(coverage, rightmost)))

    def read_hp_ratio(self, frame: FramePacket) -> float:
        crop, _ = frame.crop_norm(self.hp_roi)
        b, g, r = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
        red_mask = (r > 150) & (g < 80) & (b < 80)
        return self._measure_bar_fill(red_mask)

    def read_mp_ratio(self, frame: FramePacket) -> float:
        crop, _ = frame.crop_norm(self.mp_roi)
        b, g, r = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
        blue_mask = (b > 155) & (g < 110) & (r < 135)
        return self._measure_bar_fill(blue_mask)

    def read(self, frame: FramePacket) -> tuple[float, float]:
        return (self.read_hp_ratio(frame), self.read_mp_ratio(frame))
