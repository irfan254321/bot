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
        self.ks_melee_radius_px = float(p.get("ks_melee_radius_px", 75.0))
        self.ks_hp_damaged_threshold = float(p.get("ks_hp_damaged_threshold", 0.98))

        self.np_lo = np.array(p.get("gold_nameplate_hsv_lower", [140, 20, 58]), dtype=np.uint8)
        self.np_hi = np.array(p.get("gold_nameplate_hsv_upper", [179, 215, 255]), dtype=np.uint8)
        self.np_wrap_lo = np.array(p.get("gold_nameplate_wrap_hsv_lower", [0, 26, 58]), dtype=np.uint8)
        self.np_wrap_hi = np.array(p.get("gold_nameplate_wrap_hsv_upper", [6, 95, 110]), dtype=np.uint8)
        self.np_min_w = int(p.get("gold_nameplate_min_width", 18))
        self.np_max_w = int(p.get("gold_nameplate_max_width", 160))
        self.np_min_h = int(p.get("gold_nameplate_min_height", 5))
        self.np_max_h = int(p.get("gold_nameplate_max_height", 28))
        self.np_min_px = int(p.get("gold_nameplate_min_pixels", 10))
        self.np_min_fill = float(p.get("gold_nameplate_min_fill_ratio", 0.14))
        self.np_max_fill = float(p.get("gold_nameplate_max_fill_ratio", 0.68))
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
            r = r_end

        if best_span >= 6:
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
        """Detect other players via bright green player nameplates outside self-zone and mob HP strips."""
        sx = 1.0 if self.hud_fixed_pixel_ui else (frame_w / 1920.0)
        sy = 1.0 if self.hud_fixed_pixel_ui else (frame_h / 1080.0)

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
            return ((), ())

        kx = max(7, int(round(13.0 * sx)))
        closed = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, self._get_kernel(kx, 2))
        n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        pts: list[tuple[int, int]] = []
        boxes: list[tuple[int, int, int, int]] = []
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

        # 1. Segment pink/magenta Gold Rustler nameplates (including dark night rose-magenta H=0..6 wraparound)
        pink_main = cv2.inRange(pf_hsv, self.np_lo, self.np_hi)
        pink_wrap = cv2.inRange(pf_hsv, self.np_wrap_lo, self.np_wrap_hi)
        pink = cv2.bitwise_and(cv2.bitwise_or(pink_main, pink_wrap), hud_mask)
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
                # Require >=30% true magenta pixels (H>=140) so pure brown/red dirt or hills never match
                main_px = cv2.countNonZero(pink_main[y : y + h, x : x + w])
                if main_px < max(3, int(raw_px * 0.30)):
                    continue
                raw_nps.append((ox + x, oy + y, w, h, raw_px))

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
            dist_px = math.hypot(foot_x - feet_x, foot_y - feet_y)
            has_hp, hp_est = self._read_hp_bar_below_nameplate(pf_hsv, ox, oy, np_box, fw, fh)
            conf = min(1.0, max(0.45, raw_px / max(1.0, float(mw * mh) * 0.45)))

            # Combat-status Anti-KS (NO blind <150px discard!)
            contested = False
            contest_reason = ""
            if other_boxes:
                # (a) Pre-Click Anti-PK Overlap: clicking this mob would click directly on another player
                for pb in other_boxes:
                    if (pb[0] - pad_pk_x) <= click_x <= (pb[2] + pad_pk_x) and (pb[1] - pad_pk_y) <= click_y <= (pb[3] + pad_pk_y):
                        contested = True
                        contest_reason = "SKIP [PK_OVERLAP]"
                        break

                # (b) Another player in direct melee (< 75px) of a damaged mob (< 98% HP)
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

        # Sort safe mobs: engaged close mob first, then closest foot-point ground distance
        safe_mobs.sort(key=lambda m: (0 if (m.has_hp_bar and m.dist_to_player_px <= 220.0) else 1, m.dist_to_player_px))
        contested_mobs.sort(key=lambda m: m.dist_to_player_px)
        return (tuple(safe_mobs), tuple(contested_mobs), other_pts, other_boxes)

    def detect(self, frame: FramePacket) -> tuple[MobTarget, ...]:
        safe, _cont, _pts, _boxes = self.detect_with_context(frame)
        return safe
