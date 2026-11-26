"""Minimap radar analyzer: detects nearby red mob blips and player cyan facing cone."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig
from vikings_bot.state.models import MinimapBlip, MinimapReading


class MinimapAnalyzer:
    """Extracts nearby entity vectors and facing estimate from the North-up minimap."""

    def __init__(self, config: BotConfig) -> None:
        self.roi = config.roi("minimap")
        p = config.perception
        self.hud_fixed_pixel_ui = bool(p.get("hud_fixed_pixel_ui", False))
        self.red_lo_1 = np.array(p.get("minimap_red_hsv_lower_1", [0, 140, 120]), dtype=np.uint8)
        self.red_hi_1 = np.array(p.get("minimap_red_hsv_upper_1", [10, 255, 255]), dtype=np.uint8)
        self.red_lo_2 = np.array(p.get("minimap_red_hsv_lower_2", [170, 140, 120]), dtype=np.uint8)
        self.red_hi_2 = np.array(p.get("minimap_red_hsv_upper_2", [179, 255, 255]), dtype=np.uint8)
        self.cyan_lo = np.array(p.get("minimap_cyan_hsv_lower", [82, 95, 140]), dtype=np.uint8)
        self.cyan_hi = np.array(p.get("minimap_cyan_hsv_upper", [108, 255, 255]), dtype=np.uint8)

    def analyze(self, frame: FramePacket) -> MinimapReading:
        if self.hud_fixed_pixel_ui:
            x1 = max(0, int(round(self.roi[0] * 1920.0)))
            y1 = max(0, int(round(self.roi[1] * 1080.0)))
            x2 = min(frame.width, max(x1 + 20, int(round(self.roi[2] * 1920.0))))
            y2 = min(frame.height, max(y1 + 20, int(round(self.roi[3] * 1080.0))))
            crop = frame.image_bgr[y1:y2, x1:x2]
        else:
            crop, _ = frame.crop_norm(self.roi)
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return MinimapReading()

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cx0 = w * 0.5
        cy0 = h * 0.5
        max_r = min(w, h) * 0.43

        # 1. Red blips (nearby mobs/entities)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.red_lo_1, self.red_hi_1),
            cv2.inRange(hsv, self.red_lo_2, self.red_hi_2),
        )
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        blips: list[MinimapBlip] = []
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if not (2 <= area <= 45):
                continue
            bx, by = float(centroids[idx][0]), float(centroids[idx][1])
            dx = bx - cx0
            dy = cy0 - by  # +Y = North
            dist = math.hypot(dx, dy)
            if dist < 4.0 or dist > max_r:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            blips.append(MinimapBlip(dx_px=dx, dy_px=dy, world_angle_deg=angle, dist_px=dist))
        blips.sort(key=lambda b: b.dist_px)

        # 2. Cyan facing cone near center (annulus 3..16 px around center)
        cyan_mask = cv2.inRange(hsv, self.cyan_lo, self.cyan_hi)
        ys, xs = np.where(cyan_mask > 0)
        heading_deg: float | None = None
        heading_conf = 0.0
        if len(xs) >= 4:
            dx_arr = xs.astype(np.float32) - cx0
            dy_arr = cy0 - ys.astype(np.float32)
            r_arr = np.hypot(dx_arr, dy_arr)
            ring = (r_arr >= 2.0) & (r_arr <= 10.0)
            if int(ring.sum()) >= 3:
                # Weight outer cone pixels more strongly than central player dot
                w_arr = r_arr[ring]
                vx = float(np.average(dx_arr[ring], weights=w_arr))
                vy = float(np.average(dy_arr[ring], weights=w_arr))
                mag = math.hypot(vx, vy)
                if mag >= 1.5:
                    heading_deg = math.degrees(math.atan2(vy, vx))
                    heading_conf = min(1.0, mag / 6.0)

        return MinimapReading(
            red_blips=tuple(blips),
            heading_deg=heading_deg,
            heading_confidence=heading_conf,
        )
