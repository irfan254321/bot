"""Pure OpenCV Gold Rustler + Other-Player Anti-KS detector (<5ms CPU, 1920x1080 & 1024x768 VM)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig, roi_to_pixels
from vikings_bot.perception.hud_mask import build_playfield_hud_mask, compute_self_zone
from vikings_bot.state.models import MobTarget


def _bbox_dist(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Minimum Euclidean distance between two axis-aligned boxes (0.0 if overlapping)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0, max(ay1 - by2, by1 - ay2))
    return math.hypot(dx, dy)


class GoldRustlerDetector:
    """Lightweight Pure OpenCV Gold Rustler detector with Foot-Point distance, Torso click, HP bar & Anti-KS."""

    def __init__(self, config: BotConfig) -> None:
        self.playfield_roi = config.roi("playfield")
        self.player_center_norm = config.player_center_norm()
        p = config.perception
        self.hud_fixed_pixel_ui = bool(p.get("hud_fixed_pixel_ui", False))
        self.click_y_top_ratio = float(p.get("click_y_top_ratio", 0.35))
        self.anti_ks_enabled = bool(p.get("anti_ks_enabled", True))
        self.anti_ks_player_radius_px = float(p.get("anti_ks_player_radius_px", 240.0))
        self.anti_ks_min_untouched_hp_ratio = float(p.get("anti_ks_min_untouched_hp_ratio", 0.94))
        self.ks_melee_radius_px = float(p.get("ks_melee_radius_px", 75.0))
        self.ks_hp_damaged_threshold = float(p.get("ks_hp_damaged_threshold", self.anti_ks_min_untouched_hp_ratio))

        self.np_lo = np.array(p.get("gold_nameplate_hsv_lower", [138, 16, 48]), dtype=np.uint8)
        self.np_hi = np.array(p.get("gold_nameplate_hsv_upper", [179, 215, 255]), dtype=np.uint8)
        self.np_wrap_lo = np.array(p.get("gold_nameplate_wrap_hsv_lower", [0, 26, 58]), dtype=np.uint8)
        self.np_wrap_hi = np.array(p.get("gold_nameplate_wrap_hsv_upper", [6, 95, 110]), dtype=np.uint8)
        self.np_min_w = int(p.get("gold_nameplate_min_width", 18))
        self.np_max_w = int(p.get("gold_nameplate_max_width", 160))
        self.np_min_h = int(p.get("gold_nameplate_min_height", 5))
        self.np_max_h = int(p.get("gold_nameplate_max_height", 28))
        self.np_min_px = int(p.get("gold_nameplate_min_pixels", 10))
        self.np_min_fill = float(p.get("gold_nameplate_min_fill_ratio", 0.08))
        self.np_max_fill = float(p.get("gold_nameplate_max_fill_ratio", 0.78))
        ck = p.get("gold_nameplate_close_kernel", [14, 3])
        self.np_close_kx = max(8, int(ck[0])) if isinstance(ck, (list, tuple)) and len(ck) >= 2 else 14
        self.np_close_ky = max(2, int(ck[1])) if isinstance(ck, (list, tuple)) and len(ck) >= 2 else 3
        self.mob_body_max_height_1024 = float(p.get("mob_body_max_height_1024", 135.0))

        self.hp_lo = np.array(p.get("mob_hp_bar_hsv_lower", [48, 110, 105]), dtype=np.uint8)
        self.hp_hi = np.array(p.get("mob_hp_bar_hsv_upper", [76, 255, 255]), dtype=np.uint8)
        self.hp_min_px = int(p.get("mob_hp_bar_min_pixels", 10))

        # Friendly / other player green nameplate HSV (brighter & higher V than grass)
        self.pl_green_lo = np.array(p.get("player_nameplate_hsv_lower", [48, 90, 155]), dtype=np.uint8)
        self.pl_green_hi = np.array(p.get("player_nameplate_hsv_upper", [84, 255, 255]), dtype=np.uint8)
        # Cyan/light-blue other-player HP bar HSV (H=82..105, S>=85, V>=150, w=35..95, h=3..8 outside SELF)
        self.pl_cyan_lo = np.array(p.get("player_hp_cyan_hsv_lower", [82, 85, 150]), dtype=np.uint8)
        self.pl_cyan_hi = np.array(p.get("player_hp_cyan_hsv_upper", [105, 255, 255]), dtype=np.uint8)
        self._kernels: dict[tuple[int, int], np.ndarray] = {}

    def _get_kernel(self, kx: int, ky: int) -> np.ndarray:
        key = (int(kx), int(ky))
        k = self._kernels.get(key)
        if k is None:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, key)
            self._kernels[key] = k
        return k

    @staticmethod
    def project_body_from_nameplate(
        np_bbox: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        *,
        click_y_top_ratio: float = 0.35,
        max_body_height_1024: float = 135.0,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
        """Project 3D body_bbox, upper-torso click point, and ground foot point from nameplate bbox."""
        mx, my, mw, mh = np_bbox
        sx = frame_w / 1920.0
        sy = frame_h / 1080.0
        cx = mx + mw // 2
        depth = max(0.38, min(1.38, ((float(my) / max(1.0, float(frame_h))) - 0.14) / 0.56))
        bw = max(float(mw) * 1.10, 82.0 * sx * depth)
        max_bh = float(max_body_height_1024) if frame_w <= 1100 else (float(max_body_height_1024) * (frame_h / 768.0))
        bh = min(max_bh, max(78.0 * sy * depth, bw * 1.15))
        y1 = min(frame_h - 4, my + mh + int(round(10.0 * sy)))
        y2 = min(frame_h - 2, max(y1 + 12, int(round(y1 + bh))))
        x1 = max(0, int(round(cx - bw * 0.5)))
        x2 = min(frame_w - 1, max(x1 + 12, int(round(cx + bw * 0.5))))

        click_x = int(max(x1, min(x2, cx)))
        click_y = int(max(y1, min(y2, round(y1 + click_y_top_ratio * (y2 - y1)))))
        foot_x = click_x
        foot_y = int(y2)
        return ((x1, y1, x2, y2), (click_x, click_y), (foot_x, foot_y))

    def _read_hp_bar_below_nameplate(
        self,
        pf_hsv: np.ndarray,
        ox: int,
        oy: int,
        np_bbox: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> tuple[bool, float | None]:
        """Inspect tight strip directly below a Gold Rustler nameplate for its green target HP bar."""
        mx, my, mw, mh = np_bbox
        sx = 1.0 if self.hud_fixed_pixel_ui else (frame_w / 1920.0)
        sy = 1.0 if self.hud_fixed_pixel_ui else (frame_h / 1080.0)
        cx = mx + mw // 2
        expected_w = max(36, int(round(68.0 * sx)))
        half_w = max(expected_w // 2 + 8, mw // 2 + 6)

        lx1 = max(0, (cx - half_w) - ox)
        lx2 = min(pf_hsv.shape[1], (cx + half_w) - ox)
        ly1 = max(0, (my + mh - 1) - oy)
        ly2 = min(pf_hsv.shape[0], (my + mh + int(round(17.0 * sy))) - oy)
        if lx2 - lx1 < 12 or ly2 - ly1 < 3:
            return (False, None)

        strip = pf_hsv[ly1:ly2, lx1:lx2]
        green = cv2.inRange(strip, self.hp_lo, self.hp_hi)
        if cv2.countNonZero(green) < 3:
            return (False, None)

        green_bool = green > 0
        row_counts = green_bool.sum(axis=1)
        n_rows = int(green_bool.shape[0])
        max_bar_h = max(6, int(round(6.5 * sy)))

        # Pass A: Compact horizontal green HP bar band (2..6 rows tall, immune to stray player text/aura rows)
        dense_rows: list[bool] = [False] * n_rows
        for r in range(n_rows):
            rc = int(row_counts[r])
            if rc < 5:
                continue
            xs = np.where(green_bool[r])[0]
            span = int(xs[-1] - xs[0] + 1)
            if span >= 6 and (rc / float(span)) >= 0.50:
                dense_rows[r] = True

        best_span = 0
        best_band_rows: tuple[int, int] = (0, 0)
        best_x_end = 0
        r = 0
        while r < n_rows:
            if not dense_rows[r]:
                r += 1
                continue
            r_end = r + 1
            while r_end < n_rows and dense_rows[r_end]:
                r_end += 1
            bh = r_end - r
            if 1 <= bh <= max_bar_h:
                band = green_bool[r:r_end, :]
                band_px = int(band.sum())
                cols = band.any(axis=0)
                xs = np.where(cols)[0]
                if len(xs) > 0 and band_px >= max(8, self.hp_min_px):
                    span = int(xs[-1] - xs[0] + 1)
                    if span >= 6 and (int(cols.sum()) / float(span)) >= 0.50 and span > best_span:
                        best_span = span
                        best_band_rows = (r, r_end)
                        best_x_end = int(xs[-1])
            r = r_end

        if best_span >= 6:
            # Check if a dark/blue depleted HP bar track extends to the right of the green fill
            r1, r2 = best_band_rows
            right_x1 = min(strip.shape[1], best_x_end + 1)
            right_x2 = min(strip.shape[1], right_x1 + max(6, int(round(28.0 * sx))))
            has_depleted_track = False
            if right_x2 - right_x1 >= 4 and r2 > r1:
                r_patch = strip[r1:r2, right_x1:right_x2]
                # Depleted track in RYL is dark (V <= 42) or dark-blue (H=100..140, S>=50, V<=145)
                dark_or_blue = (r_patch[:, :, 2] <= 42) | (
                    (r_patch[:, :, 0] >= 100) & (r_patch[:, :, 0] <= 140) & (r_patch[:, :, 1] >= 50) & (r_patch[:, :, 2] <= 145)
                )
                has_depleted_track = float(dark_or_blue.mean()) >= 0.35
            if (not has_depleted_track) and best_span >= int(round(35.0 * sx)):
                ratio = 1.0
            else:
                ratio = max(0.03, min(1.0, float(best_span) / float(expected_w)))
            return (True, ratio)

        # Pass B: Ultra-low HP sliver (1..6 green px/row stacked across 2..5 rows inside dark HP bar border)
        v_ch = strip[:, :, 2]
        r = 0
        while r < n_rows:
            if not (1 <= int(row_counts[r]) <= 6):
                r += 1
                continue
            r_end = r + 1
            while r_end < n_rows and (1 <= int(row_counts[r_end]) <= 6):
                r_end += 1
            bh = r_end - r
            if 2 <= bh <= max(5, int(round(5.5 * sy))):
                band = green_bool[r:r_end, :]
                xs = np.where(band.any(axis=0))[0]
                if len(xs) > 0:
                    span = int(xs[-1] - xs[0] + 1)
                    if span <= max(7, int(round(9.0 * sx))):
                        top_a = max(0, r - 3)
                        top_b = max(top_a + 1, r)
                        has_dark_border = any(
                            float((v_ch[rb] <= 75).mean()) >= 0.60 for rb in range(top_a, top_b)
                        )
                        if has_dark_border:
                            ratio = max(0.02, min(0.18, float(span) / float(expected_w)))
                            return (True, ratio)
            r = r_end

        return (False, None)

    def _detect_other_players(
        self,
        pf_hsv: np.ndarray,
        hud_mask: np.ndarray,
        ox: int,
        oy: int,
        frame_w: int,
        frame_h: int,
        self_box: tuple[int, int, int, int],
        mob_nameplates: list[tuple[int, int, int, int]],
    ) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int, int], ...]]:
        """Detect other players via cyan/light-blue HP bars AND bright green player nameplates outside SELF."""
        sx = 1.0 if self.hud_fixed_pixel_ui else (frame_w / 1920.0)
        sy = 1.0 if self.hud_fixed_pixel_ui else (frame_h / 1080.0)
        sx_1024 = 1.0 if self.hud_fixed_pixel_ui else (frame_w / 1024.0)
        sy_1024 = 1.0 if self.hud_fixed_pixel_ui else (frame_h / 768.0)

        pts: list[tuple[int, int]] = []
        boxes: list[tuple[int, int, int, int]] = []

        # Pass A: Cyan / light-blue player HP bars (HSV H=82..105, S>=85, V>=150, w=35..95, h=3..8) outside SELF
        cmask = cv2.bitwise_and(cv2.inRange(pf_hsv, self.pl_cyan_lo, self.pl_cyan_hi), hud_mask)
        # Mask SELF zone (x=435..590, y=305..465 at 1024x768, scaled) + self_box
        for ex1, ey1, ex2, ey2 in (
            self_box,
            (
                int(round(435.0 * sx_1024)),
                int(round(305.0 * sy_1024)),
                int(round(590.0 * sx_1024)),
                int(round(465.0 * sy_1024)),
            ),
        ):
            lx1, ly1 = max(0, ex1 - ox), max(0, ey1 - oy)
            lx2, ly2 = min(cmask.shape[1], ex2 - ox), min(cmask.shape[0], ey2 - oy)
            if lx2 > lx1 and ly2 > ly1:
                cmask[ly1:ly2, lx1:lx2] = 0

        min_cw, max_cw = int(round(35.0 * sx_1024)), int(round(95.0 * sx_1024))
        min_ch, max_ch = max(3, int(round(3.0 * sy_1024))), max(8, int(round(8.0 * sy_1024)))
        if cv2.countNonZero(cmask) >= max(45, int(round(70.0 * sx_1024 * sy_1024))):
            n_c, _lc, st_c, _ = cv2.connectedComponentsWithStats(cmask, connectivity=8)
            for idx in range(1, n_c):
                x = int(st_c[idx, cv2.CC_STAT_LEFT])
                y = int(st_c[idx, cv2.CC_STAT_TOP])
                w = int(st_c[idx, cv2.CC_STAT_WIDTH])
                h = int(st_c[idx, cv2.CC_STAT_HEIGHT])
                if not (min_cw <= w <= max_cw and min_ch <= h <= max_ch):
                    continue
                if (int(st_c[idx, cv2.CC_STAT_AREA]) / max(1.0, float(w * h))) < 0.50:
                    continue
                fx = ox + x + w // 2
                fy = oy + y + h // 2
                if any(abs(fx - px) <= 45 * sx_1024 and abs(fy - py) <= 45 * sy_1024 for px, py in pts):
                    continue
                pts.append((fx, fy))
                half_bw = int(round(38.0 * sx_1024))
                bh = int(round(115.0 * sy_1024))
                boxes.append((max(0, fx - half_bw), max(0, fy), min(frame_w - 1, fx + half_bw), min(frame_h - 1, fy + bh)))

        # Pass B: Bright green player nameplates outside self-zone and mob HP strips
        gmask = cv2.inRange(pf_hsv, self.pl_green_lo, self.pl_green_hi)
        gmask = cv2.bitwise_and(gmask, hud_mask)

        # Erase thin strips directly below detected Gold Rustler nameplates so mob HP bars aren't players
        for mx, my, mw, mh in mob_nameplates:
            ex1 = max(0, (mx - int(round(18.0 * sx))) - ox)
            ex2 = min(gmask.shape[1], (mx + mw + int(round(18.0 * sx))) - ox)
            ey1 = max(0, (my - 2) - oy)
            ey2 = min(gmask.shape[0], (my + mh + int(round(24.0 * sy))) - oy)
            if ex2 > ex1 and ey2 > ey1:
                gmask[ey1:ey2, ex1:ex2] = 0

        # Erase self-character nameplate zone
        sx1, sy1, sx2, sy2 = self_box
        lx1, ly1 = max(0, sx1 - ox), max(0, sy1 - oy)
        lx2, ly2 = min(gmask.shape[1], sx2 - ox), min(gmask.shape[0], sy2 - oy)
        if lx2 > lx1 and ly2 > ly1:
            gmask[ly1:ly2, lx1:lx2] = 0

        min_raw_px = max(12, int(round(18.0 * sx * sy)))
        if cv2.countNonZero(gmask) < min_raw_px:
            return (tuple(pts), tuple(boxes))

        kx = max(7, int(round(13.0 * sx)))
        closed = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, self._get_kernel(kx, 2))
        n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        min_w = max(18, int(round(24.0 * sx)))
        max_w = int(round(190.0 * sx))
        min_h = max(4, int(round(5.0 * sy)))
        max_h = max(18, int(round(24.0 * sy)))

        for idx in range(1, n_lbl):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if not (min_w <= w <= max_w and min_h <= h <= max_h):
                continue
            if (w / max(1.0, float(h))) < 1.8:
                continue
            raw_px = int((gmask[y : y + h, x : x + w] > 0).sum())
            if raw_px < max(12, int(round(18.0 * sx * sy))):
                continue

            # Player nameplates sit on dark outline / sky / grass without solid neon fill
            fill_ratio = raw_px / max(1.0, float(w * h))
            if fill_ratio > 0.78:
                continue

            fx = ox + x + w // 2
            fy = oy + y + h // 2
            # Deduplicate stacked guild + player name lines belonging to the same player
            if any(abs(fx - px) <= 45 * sx and abs(fy - py) <= 42 * sy for px, py in pts):
                continue

            pts.append((fx, fy))
            half_bw = int(round(38.0 * sx))
            bh = int(round(125.0 * sy))
            bx1 = max(0, fx - half_bw)
            by1 = max(0, oy + y + h)
            bx2 = min(frame_w - 1, fx + half_bw)
            by2 = min(frame_h - 1, by1 + bh)
            boxes.append((bx1, by1, bx2, by2))

        return (tuple(pts), tuple(boxes))

    def detect_with_context(
        self,
        frame: FramePacket,
        shared_playfield_hsv: np.ndarray | None = None,
    ) -> tuple[
        tuple[MobTarget, ...],
        tuple[MobTarget, ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int, int, int], ...],
    ]:
        """Detect safe Gold Rustlers, KS-contested Gold Rustlers, and other players.

        Returns `(safe_mobs, contested_mobs, other_player_pts, other_player_boxes)`.
        """
        fw, fh = frame.width, frame.height
        if fw < 64 or fh < 64:
            return ((), (), (), ())

        pf_rect = roi_to_pixels(self.playfield_roi, fw, fh)
        ox, oy, px2, py2 = pf_rect
        if shared_playfield_hsv is not None:
            pf_hsv = shared_playfield_hsv
        else:
            pf_crop = frame.image_bgr[oy:py2, ox:px2]
            if pf_crop.size == 0:
                return ((), (), (), ())
            pf_hsv = cv2.cvtColor(pf_crop, cv2.COLOR_BGR2HSV)

        hud_mask = build_playfield_hud_mask(fw, fh, pf_rect, hud_fixed_pixel_ui=self.hud_fixed_pixel_ui)
        (feet_x, feet_y), self_box = compute_self_zone(
            fw, fh, player_center_norm=self.player_center_norm, hud_fixed_pixel_ui=self.hud_fixed_pixel_ui
        )

        sx = 1.0 if self.hud_fixed_pixel_ui else (fw / 1920.0)
        sy = 1.0 if self.hud_fixed_pixel_ui else (fh / 1080.0)

        # 1. Segment pink/magenta Gold Rustler nameplates (daylight-strict + dark night rose-magenta wraparound)
        pink_main = cv2.inRange(pf_hsv, self.np_lo, self.np_hi)
        # Check playfield brightness: in bright daylight, disable warm brown/red wraparound so grass never triggers!
        med_v = float(np.median(pf_hsv[::8, ::8, 2])) if pf_hsv.size > 0 else 70.0
        is_night = med_v < 72.0
        if is_night:
            pink_wrap = cv2.inRange(pf_hsv, self.np_wrap_lo, self.np_wrap_hi)
            pink = cv2.bitwise_and(cv2.bitwise_or(pink_main, pink_wrap), hud_mask)
        else:
            pink = cv2.bitwise_and(pink_main, hud_mask)
        min_w = max(14, int(round(self.np_min_w * sx)))
        max_w = int(round(self.np_max_w * sx))
        min_h = max(4, int(round(self.np_min_h * sy)))
        max_h = max(18, int(round(self.np_max_h * sy)))
        min_px = max(8, int(round(self.np_min_px * sx * sy)))

        raw_nps: list[tuple[int, int, int, int, int]] = []
        if cv2.countNonZero(pink) >= min_px:
            kx = max(10, int(round(float(self.np_close_kx) * sx)))
            ky = max(2, int(round(float(self.np_close_ky) * sy)))
            closed = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, self._get_kernel(kx, ky))
            n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

            for idx in range(1, n_lbl):
                x = int(stats[idx, cv2.CC_STAT_LEFT])
                y = int(stats[idx, cv2.CC_STAT_TOP])
                w = int(stats[idx, cv2.CC_STAT_WIDTH])
                h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                if not (min_w <= w <= max_w and min_h <= h <= max_h):
                    continue
                if (w / max(1.0, float(h))) < 1.55:
                    continue
                raw_px = cv2.countNonZero(pink[y : y + h, x : x + w])
                if raw_px < min_px:
                    continue
                # Validate text stroke density (fill_ratio 0.14..0.68) to reject solid hill/horizon smudges
                fill_ratio = raw_px / max(1.0, float(w * h))
                if not (self.np_min_fill <= fill_ratio <= self.np_max_fill):
                    continue
                # Require >=18px dense magenta pixels in daylight (or >=30% true magenta at night) so empty grass never triggers
                main_px = cv2.countNonZero(pink_main[y : y + h, x : x + w])
                min_dense_magenta = max(14, int(round(18.0 * sx * sy)))
                if is_night:
                    if main_px < max(3, int(raw_px * 0.30)):
                        continue
                else:
                    if main_px < max(min_dense_magenta, int(raw_px * 0.50)):
                        continue
                raw_nps.append((ox + x, oy + y, w, h, raw_px))

        # 1b. Standalone horizontal green HP-bar detector (works without magenta nameplate!)
        #     HSV H=40..85, S>=110, V>=110, width 32..115px, height 3..9px -> click torso at (bar_cx, bar_y + 42)!
        #     Tight SELF exclusion box (x=472..552, y=350..432 at 1024x768, scaled) so close mobs are NEVER blocked!
        hp_bar_mask = cv2.bitwise_and(
            cv2.inRange(pf_hsv, np.array([40, 110, 110], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
            hud_mask,
        )
        sx_1024 = 1.0 if self.hud_fixed_pixel_ui else (fw / 1024.0)
        sy_1024 = 1.0 if self.hud_fixed_pixel_ui else (fh / 768.0)
        self_excl_x1 = int(round(472.0 * sx_1024))
        self_excl_y1 = int(round(350.0 * sy_1024))
        self_excl_x2 = int(round(552.0 * sx_1024))
        self_excl_y2 = int(round(432.0 * sy_1024))
        hp_bar_mask[
            max(0, self_excl_y1 - oy) : min(hp_bar_mask.shape[0], self_excl_y2 - oy),
            max(0, self_excl_x1 - ox) : min(hp_bar_mask.shape[1], self_excl_x2 - ox),
        ] = 0

        hp_bar_click_y_by_np: dict[tuple[int, int, int, int], int] = {}
        if cv2.countNonZero(hp_bar_mask) >= max(45, int(round(64.0 * sx_1024 * sy_1024))):
            n_hb, _lhb, st_hb, _ = cv2.connectedComponentsWithStats(hp_bar_mask, connectivity=8)
            min_hbw, max_hbw = int(round(32.0 * sx_1024)), int(round(115.0 * sx_1024))
            min_hbh, max_hbh = max(3, int(round(3.0 * sy_1024))), max(9, int(round(9.0 * sy_1024)))
            for idx in range(1, n_hb):
                hx = int(st_hb[idx, cv2.CC_STAT_LEFT])
                hy = int(st_hb[idx, cv2.CC_STAT_TOP])
                hw = int(st_hb[idx, cv2.CC_STAT_WIDTH])
                hh = int(st_hb[idx, cv2.CC_STAT_HEIGHT])
                if not (min_hbw <= hw <= max_hbw and min_hbh <= hh <= max_hbh):
                    continue
                if (int(st_hb[idx, cv2.CC_STAT_AREA]) / max(1.0, float(hw * hh))) < 0.50:
                    continue
                gx, gy = ox + hx, oy + hy
                if any(abs(gx - nx) <= 45 * sx_1024 and abs(gy - (ny + nh)) <= 26 * sy_1024 for nx, ny, _nw, nh, _ in raw_nps):
                    continue
                syn_np = (gx, max(oy, gy - int(round(14.0 * sy_1024))), hw, max(8, int(round(9.0 * sy_1024))))
                raw_nps.append((*syn_np, hw * 4))
                hp_bar_click_y_by_np[syn_np] = min(fh - 2, int(round(gy + 42.0 * sy_1024)))

        np_bboxes = [(x, y, w, h) for x, y, w, h, _ in raw_nps]

        # 2. Detect other players for combat-status Anti-KS & Anti-PK overlap guard
        other_pts, other_boxes = self._detect_other_players(
            pf_hsv, hud_mask, ox, oy, fw, fh, self_box, np_bboxes
        )

        # 3. Project body boxes, foot distances, upper-torso click points, HP bars, and Anti-KS status
        safe_mobs: list[MobTarget] = []
        contested_mobs: list[MobTarget] = []
        melee_ks_r = self.ks_melee_radius_px * (fw / 1920.0)
        pad_pk_x = 24.0 * (fw / 1920.0)
        pad_pk_y = 16.0 * (fh / 1080.0)

        for mx, my, mw, mh, raw_px in raw_nps:
            np_box = (mx, my, mw, mh)
            body_bbox, (click_x, click_y), (foot_x, foot_y) = self.project_body_from_nameplate(
                np_box,
                fw,
                fh,
                click_y_top_ratio=self.click_y_top_ratio,
                max_body_height_1024=self.mob_body_max_height_1024,
            )
            if np_box in hp_bar_click_y_by_np:
                click_y = hp_bar_click_y_by_np[np_box]
            # Reject any candidate whose click point falls inside the expanded SELF exclusion zone (425..600, 300..555)
            if self_excl_x1 <= click_x <= self_excl_x2 and self_excl_y1 <= click_y <= self_excl_y2:
                continue
            dist_px = math.hypot(foot_x - feet_x, foot_y - feet_y)
            has_hp, hp_est = self._read_hp_bar_below_nameplate(pf_hsv, ox, oy, np_box, fw, fh)
            conf = min(1.0, max(0.45, raw_px / max(1.0, float(mw * mh) * 0.45)))

            # Anti-KS & Player Avoidance:
            # Reject monsters within <=240px of other_players OR already damaged (hp_ratio < 0.94)!
            contested = False
            contest_reason = ""
            if self.anti_ks_enabled and has_hp and hp_est is not None and hp_est < self.anti_ks_min_untouched_hp_ratio:
                contested = True
                contest_reason = f"SKIP [KS_HP<{int(self.anti_ks_min_untouched_hp_ratio * 100)}%]"
            if not contested and other_pts and self.anti_ks_enabled:
                player_avoid_r = self.anti_ks_player_radius_px * sx_1024
                if any(math.hypot(click_x - opx, click_y - opy) <= player_avoid_r for opx, opy in other_pts):
                    contested = True
                    contest_reason = "SKIP [KS_PLAYER]"
            if not contested and other_boxes:
                # (a) Pre-Click Anti-PK Overlap: clicking this mob would click directly on another player
                for pb in other_boxes:
                    if (pb[0] - pad_pk_x) <= click_x <= (pb[2] + pad_pk_x) and (pb[1] - pad_pk_y) <= click_y <= (pb[3] + pad_pk_y):
                        contested = True
                        contest_reason = "SKIP [PK_OVERLAP]"
                        break

                # (b) Another player in direct melee (< 75px) of a damaged mob
                if not contested and has_hp and (hp_est is None or hp_est < self.ks_hp_damaged_threshold):
                    min_p_dist = min(_bbox_dist(pb, body_bbox) for pb in other_boxes)
                    if min_p_dist < melee_ks_r:
                        contested = True
                        contest_reason = "SKIP [KS_MELEE]"
                    elif dist_px > 260.0 * (fw / 1920.0) and min_p_dist < 160.0 * (fw / 1920.0):
                        contested = True
                        contest_reason = "SKIP [KS_OTHER]"

            mob = MobTarget(
                nameplate_bbox=np_box,
                body_bbox=body_bbox,
                click_x=click_x,
                click_y=click_y,
                foot_x=foot_x,
                foot_y=foot_y,
                dist_to_player_px=float(dist_px),
                has_hp_bar=has_hp,
                hp_ratio_est=hp_est,
                confidence=round(conf, 3),
                contested=contested,
                contest_reason=contest_reason,
            )
            if contested:
                contested_mobs.append(mob)
            else:
                safe_mobs.append(mob)

        # 3b. Close-range monster silhouette detector (for unhovered close mobs without nameplate/HP bar, e.g. ref_close_mobs.jpg)
        if not is_night:
            sil_mobs = self._detect_close_silhouette_mobs(
                pf_hsv,
                hud_mask,
                ox,
                oy,
                fw,
                fh,
                feet_x,
                feet_y,
                (self_excl_x1, self_excl_y1, self_excl_x2, self_excl_y2),
                safe_mobs + contested_mobs,
                other_pts,
                other_boxes,
            )
            for sm in sil_mobs:
                if sm.contested:
                    contested_mobs.append(sm)
                else:
                    safe_mobs.append(sm)

        # Sort safe mobs: engaged close mob first, then closest foot-point ground distance
        safe_mobs.sort(key=lambda m: (0 if (m.has_hp_bar and m.dist_to_player_px <= 220.0) else 1, m.dist_to_player_px))
        contested_mobs.sort(key=lambda m: m.dist_to_player_px)
        return (tuple(safe_mobs), tuple(contested_mobs), other_pts, other_boxes)

    def _detect_close_silhouette_mobs(
        self,
        pf_hsv: np.ndarray,
        hud_mask: np.ndarray,
        ox: int,
        oy: int,
        fw: int,
        fh: int,
        feet_x: int,
        feet_y: int,
        self_excl: tuple[int, int, int, int],
        existing_mobs: list[MobTarget],
        other_pts: tuple[tuple[int, int], ...],
        other_boxes: tuple[tuple[int, int, int, int], ...],
    ) -> list[MobTarget]:
        """Detect close-range non-grass metallic/dark-bronze monster silhouettes (area 2800..32000px^2, y=235..590)."""
        sx = 1.0 if self.hud_fixed_pixel_ui else (fw / 1024.0)
        sy = 1.0 if self.hud_fixed_pixel_ui else (fh / 768.0)
        y1 = max(0, int(round(235.0 * sy)) - oy)
        y2 = min(pf_hsv.shape[0], int(round(590.0 * sy)) - oy)
        if y2 - y1 < 32 or pf_hsv.shape[1] < 32:
            return []

        # Half-resolution stride for <0.4ms CPU cost
        sub = pf_hsv[y1:y2:2, ::2]
        hm_s = hud_mask[y1:y2:2, ::2]
        h_ch, s_ch, v_ch = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]

        # Require olive-green grass to be the dominant ground texture (>=35%) so non-field scenes never trigger
        grass = (h_ch >= 24) & (h_ch <= 46) & (s_ch >= 45) & (v_ch >= 25)
        valid_s = hm_s > 0
        if not bool(np.any(valid_s)) or float(grass[valid_s].mean()) < 0.35:
            return []

        # Non-grass metallic gray / dark-bronze monster body pixels
        body = (~grass) & (s_ch <= 115) & (v_ch >= 14) & (v_ch <= 185) & valid_s
        mask = (body.astype(np.uint8)) * 255

        # Mask tight SELF zone (472..552, 350..432)
        sx1, sy1, sx2, sy2 = self_excl
        lx1 = max(0, (sx1 - ox) // 2)
        lx2 = min(mask.shape[1], (sx2 - ox + 1) // 2)
        ly1 = max(0, (sy1 - oy - y1) // 2)
        ly2 = min(mask.shape[0], (sy2 - oy - y1 + 1) // 2)
        if lx2 > lx1 and ly2 > ly1:
            mask[ly1:ly2, lx1:lx2] = 0

        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._get_kernel(7, 7))
        n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        # Area in 0.5x^2 space (1/4 of full-res area: 2800..32000 -> 700..8000)
        area_scale = sx * sy * 0.25
        min_a = max(300, int(round(2800.0 * area_scale)))
        max_a = int(round(32000.0 * area_scale))
        min_w = max(20, int(round(45.0 * sx * 0.5)))
        max_w = int(round(240.0 * sx * 0.5))
        min_h = max(25, int(round(60.0 * sy * 0.5)))
        max_h = int(round(280.0 * sy * 0.5))

        out: list[MobTarget] = []
        player_avoid_r = self.anti_ks_player_radius_px * sx
        for idx in range(1, n_lbl):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if not (min_a <= area <= max_a):
                continue
            bx = int(stats[idx, cv2.CC_STAT_LEFT])
            by = int(stats[idx, cv2.CC_STAT_TOP])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if not (min_w <= bw <= max_w and min_h <= bh <= max_h):
                continue
            asp = bw / max(1.0, float(bh))
            if not (0.40 <= asp <= 2.2):
                continue
            # Require mix of dark-bronze and metallic gray tones (rejects flat dirt patches / shadows)
            pv = v_ch[by : by + bh, bx : bx + bw]
            ps = s_ch[by : by + bh, bx : bx + bw]
            pb = body[by : by + bh, bx : bx + bw]
            if not bool(np.any(pb)):
                continue
            dark_r = float((pv[pb] <= 65).mean())
            metal_r = float(((ps[pb] <= 65) & (pv[pb] >= 75)).mean())
            if dark_r < 0.12 or metal_r < 0.08:
                continue

            fx1 = ox + bx * 2
            fy1 = oy + y1 + by * 2
            fx2 = min(fw - 1, ox + (bx + bw) * 2)
            fy2 = min(fh - 1, oy + y1 + (by + bh) * 2)
            cx = (fx1 + fx2) // 2
            cy = fy1 + int(round(0.42 * (fy2 - fy1)))
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                continue
            if any(math.hypot(cx - m.click_x, cy - m.click_y) <= 95.0 * sx for m in existing_mobs):
                continue
            if any(pbx1 <= cx <= pbx2 and pby1 <= cy <= pby2 for pbx1, pby1, pbx2, pby2 in other_boxes):
                continue

            contested = self.anti_ks_enabled and any(
                math.hypot(cx - opx, cy - opy) <= player_avoid_r for opx, opy in other_pts
            )
            dist_px = math.hypot(cx - feet_x, fy2 - feet_y)
            out.append(
                MobTarget(
                    nameplate_bbox=(fx1, max(0, fy1 - 12), max(16, fx2 - fx1), 10),
                    body_bbox=(fx1, fy1, fx2, fy2),
                    click_x=cx,
                    click_y=cy,
                    foot_x=cx,
                    foot_y=fy2,
                    dist_to_player_px=float(dist_px),
                    has_hp_bar=False,
                    hp_ratio_est=None,
                    confidence=0.72,
                    contested=contested,
                    contest_reason="SKIP [KS_PLAYER]" if contested else "",
                )
            )
        return out

    def detect(self, frame: FramePacket) -> tuple[MobTarget, ...]:
        safe, _cont, _pts, _boxes = self.detect_with_context(frame)
        return safe
