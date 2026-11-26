"""Fast column-wise HP and MP bar reader resilient to centered text overlays."""

from __future__ import annotations

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig


class HudBarReader:
    """Reads normalized HP (red) and MP (blue/purple) fill ratios from bottom-left HUD.

    Supports both proportional 1920x1080 ROIs and 1:1 fixed-pixel bottom-left anchored ROIs
    for VMware 1024x768 (`hud_fixed_pixel_ui=True`), with automatic small-shift fallback so
    VMware window borders or slight DPI offsets never cause false 0% HP readings.
    """

    def __init__(self, config: BotConfig) -> None:
        self.hp_roi = config.roi("hp_bar")
        self.mp_roi = config.roi("mp_bar")
        self.hud_fixed_pixel_ui = bool(config.perception.get("hud_fixed_pixel_ui", False))

    @staticmethod
    def _red_mask(crop_bgr: np.ndarray) -> np.ndarray:
        b = crop_bgr[:, :, 0]
        g = crop_bgr[:, :, 1]
        r = crop_bgr[:, :, 2]
        max_gb = np.maximum(g, b).astype(np.int16)
        return (r > 135) & (g < 85) & (b < 85) & ((r.astype(np.int16) - max_gb) >= 65)

    @staticmethod
    def _blue_mask(crop_bgr: np.ndarray) -> np.ndarray:
        b = crop_bgr[:, :, 0]
        g = crop_bgr[:, :, 1]
        r = crop_bgr[:, :, 2]
        return (b > 145) & (g < 115) & (r < 140) & ((b.astype(np.int16) - g.astype(np.int16)) >= 35)

    @staticmethod
    def _fixed_bottom_left_rect(
        roi_norm: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int]:
        """Convert a 1920x1080-calibrated bottom-left ROI into 1:1 fixed pixels anchored at (0, frame_h)."""
        ref_x1 = int(round(roi_norm[0] * 1920.0))
        ref_x2 = int(round(roi_norm[2] * 1920.0))
        # Calibrated from vm_1024_raw.jpg (HP red: x=18..98, y=693..708; MP blue: x=120..199, y=695..707)
        top_from_bot = int(round((1.0 - roi_norm[1]) * 1080.0)) + 3
        bot_from_bot = max(40, int(round((1.0 - roi_norm[3]) * 1080.0)) + 1)
        x1 = max(0, min(frame_w - 1, ref_x1))
        x2 = max(x1 + 1, min(frame_w, ref_x2))
        y1 = max(0, min(frame_h - 1, frame_h - top_from_bot))
        y2 = max(y1 + 1, min(frame_h, frame_h - bot_from_bot))
        return (x1, y1, x2, y2)

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

    def _read_bar_ratio(
        self,
        frame: FramePacket,
        roi_norm: tuple[float, float, float, float],
        mask_fn,
    ) -> float:
        fw, fh = frame.width, frame.height
        if fw < 64 or fh < 64:
            return 0.0

        if not self.hud_fixed_pixel_ui:
            crop, _ = frame.crop_norm(roi_norm)
            return self._measure_bar_fill(mask_fn(crop))

        # 1. Exact 1:1 fixed-pixel bottom-left bar box (RYL 1024x768: HP=(18,699,98,711), MP=(120,699,199,711))
        x1, y1, x2, y2 = self._fixed_bottom_left_rect(roi_norm, fw, fh)
        bar_w = max(60, x2 - x1)
        primary = frame.image_bgr[y1:y2, x1:x2]
        ratio = self._measure_bar_fill(mask_fn(primary))
        if ratio > 0.0:
            return ratio

        # 2. Adaptive bottom-left shift fallback (tolerates +-24px VMware window border / DPI offsets)
        #    When truly dead (0% HP), there are zero red bar pixels anywhere in this panel -> returns 0.0!
        sx1 = max(0, x1 - 14)
        sx2 = min(fw, x2 + 22)
        sy1 = max(0, fh - 110)
        sy2 = min(fh, fh - 26)
        if (sy2 - sy1) < 6 or (sx2 - sx1) < 30:
            return 0.0
        band_mask = mask_fn(frame.image_bgr[sy1:sy2, sx1:sx2])
        row_hits = np.where(band_mask.sum(axis=1) >= 4)[0]
        if row_hits.size < 2:
            return 0.0
        splits = np.where(np.diff(row_hits) > 2)[0] + 1
        runs = np.split(row_hits, splits)
        best_run = max(runs, key=lambda r: int(band_mask[r[0] : r[-1] + 1, :].sum()))
        if best_run.size < 2:
            return 0.0
        ry1 = int(best_run[0])
        ry2 = min(band_mask.shape[0], int(best_run[-1]) + 1)
        sub = band_mask[ry1:ry2, :]
        if int(sub.sum()) < 8:
            return 0.0
        cols = np.where(sub.any(axis=0))[0]
        if cols.size == 0:
            return 0.0
        # RYL bars drain right-to-left, so leftmost filled pixel anchors the 0% edge of the canonical bar_w slot
        left_x = int(cols[0])
        right_x = min(band_mask.shape[1], left_x + bar_w)
        slot = np.zeros((ry2 - ry1, bar_w), dtype=bool)
        w_copy = min(bar_w, right_x - left_x)
        slot[:, :w_copy] = sub[:, left_x : left_x + w_copy]
        return self._measure_bar_fill(slot)

    def read_hp_ratio(self, frame: FramePacket) -> float:
        return self._read_bar_ratio(frame, self.hp_roi, self._red_mask)

    def read_mp_ratio(self, frame: FramePacket) -> float:
        return self._read_bar_ratio(frame, self.mp_roi, self._blue_mask)

    def read(self, frame: FramePacket) -> tuple[float, float]:
        return (self.read_hp_ratio(frame), self.read_mp_ratio(frame))
