"""Pure OpenCV Dual-Cue Gold Bag detector + Dead Corpse ('Mayat') rejector + Chat pickup verifier."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig, roi_to_pixels
from vikings_bot.perception.hud_mask import build_playfield_hud_mask, compute_self_zone
from vikings_bot.state.models import LootDrop, MobTarget


class GoldBagDetector:
    """Detects dropped Gold Bags while rejecting standing mobs, players, HUD, and prone corpses ('mayat')."""

    def __init__(self, config: BotConfig) -> None:
        self.playfield_roi = config.roi("playfield")
        self.loot_roi = config.roi("loot_area") if "loot_area" in config.rois else self.playfield_roi
        self.chat_roi = config.roi("chat_log") if "chat_log" in config.rois else (0.70, 0.78, 0.99, 0.97)
        self.player_center_norm = config.player_center_norm()

        p = config.perception
        self.hud_fixed_pixel_ui = bool(p.get("hud_fixed_pixel_ui", False))
        self.glow_lo = np.array(p.get("loot_glow_hsv_lower", [10, 105, 155]), dtype=np.uint8)
        self.glow_hi = np.array(p.get("loot_glow_hsv_upper", [25, 255, 255]), dtype=np.uint8)
        self.dark_body_v_max = int(p.get("loot_dark_body_v_max", 78))
        self.min_dark_ratio = float(p.get("loot_min_dark_ratio", 0.25))
        self.min_blob_px = int(p.get("loot_min_blob_pixels", 3))
        self.max_blob_px = int(p.get("loot_max_blob_pixels", 120))
        self.pouch_v_max = int(p.get("loot_pouch_v_max", 48))
        self.pouch_min_area = int(p.get("loot_pouch_min_area", 45))
        self.pouch_max_area = int(p.get("loot_pouch_max_area", 480))
        self.pouch_warm_min_px = int(p.get("loot_pouch_warm_min_px", 8))
        self.cluster_radius_px = float(p.get("loot_cluster_radius_px", 20.0))
        self.max_dist_px = float(p.get("loot_max_dist_px", 580.0))

        tmpl_rel = str(p.get("picked_up_template", "assets/ui_templates/picked_up_prefix.png"))
        tmpl_path = config.resolve_path(tmpl_rel)
        self.pickup_tmpl: np.ndarray | None = (
            cv2.imread(str(tmpl_path), cv2.IMREAD_COLOR) if tmpl_path.exists() else None
        )
        self._last_chat_sig: bytes | None = None
        self._blacklist: list[tuple[float, float, float]] = []  # (expire_ts, x, y)
        self.last_corpse_boxes: tuple[tuple[int, int, int, int], ...] = ()
        self._corpse_kernels: dict[tuple[int, int], np.ndarray] = {}
        self._warm_lo = np.array([6, 95, 85], dtype=np.uint8)
        self._warm_hi = np.array([26, 255, 255], dtype=np.uint8)
        self._sur_warm_lo = np.array([8, 80, 70], dtype=np.uint8)
        self._sur_warm_hi = np.array([28, 255, 255], dtype=np.uint8)
        self._silver_lo = np.array([0, 0, 130], dtype=np.uint8)
        self._silver_hi = np.array([179, 55, 255], dtype=np.uint8)
        self._tunic_lo = np.array([68, 65, 45], dtype=np.uint8)
        self._tunic_hi = np.array([135, 255, 255], dtype=np.uint8)
        self._hp_green_lo = np.array([46, 105, 100], dtype=np.uint8)
        self._hp_green_hi = np.array([78, 255, 255], dtype=np.uint8)
        self._hp_red_lo_1 = np.array([0, 140, 120], dtype=np.uint8)
        self._hp_red_hi_1 = np.array([7, 255, 255], dtype=np.uint8)
        self._hp_red_lo_2 = np.array([172, 140, 120], dtype=np.uint8)
        self._hp_red_hi_2 = np.array([179, 255, 255], dtype=np.uint8)
        self.hp_bar_guard_below_px = int(p.get("loot_hp_bar_guard_below_px", 110))
        self.horizon_min_y_px = int(p.get("loot_horizon_min_y_px", 230))

    def _mask_below_hp_bar_strips(
        self,
        hsv_crop: np.ndarray,
        valid: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> None:
        """Mask monster body columns (up to +110px below) under any horizontal green/red HP bar strip.

        Prevents golden chest/armor gleam on a monster from being mistaken for a Gold Bag even when
        dusk/night lighting hides the monster's magenta nameplate.
        """
        sx = 1.0 if self.hud_fixed_pixel_ui else scale_x
        sy = 1.0 if self.hud_fixed_pixel_ui else scale_y
        min_w = max(14, int(round(18.0 * sx)))
        max_w = int(round(115.0 * sx))
        min_h = max(2, int(round(2.0 * sy)))
        max_h = max(7, int(round(8.0 * sy)))
        pad_x = int(round(22.0 * sx))
        pad_below = int(round(float(self.hp_bar_guard_below_px) * sy))

        g_mask = cv2.bitwise_and(cv2.inRange(hsv_crop, self._hp_green_lo, self._hp_green_hi), valid)
        r_mask_1 = cv2.inRange(hsv_crop, self._hp_red_lo_1, self._hp_red_hi_1)
        r_mask_2 = cv2.inRange(hsv_crop, self._hp_red_lo_2, self._hp_red_hi_2)
        r_mask = cv2.bitwise_and(cv2.bitwise_or(r_mask_1, r_mask_2), valid)

        vh, vw = valid.shape[:2]
        for bar_mask in (g_mask, r_mask):
            if cv2.countNonZero(bar_mask) < (min_w * min_h):
                continue
            n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(bar_mask, connectivity=8)
            for idx in range(1, n_lbl):
                w = int(stats[idx, cv2.CC_STAT_WIDTH])
                h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                if not (min_w <= w <= max_w and min_h <= h <= max_h):
                    continue
                if (w / max(1.0, float(h))) < 3.0:
                    continue
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if (area / max(1.0, float(w * h))) < 0.55:
                    continue
                x = int(stats[idx, cv2.CC_STAT_LEFT])
                y = int(stats[idx, cv2.CC_STAT_TOP])
                ax1 = max(0, x - pad_x)
                ax2 = min(vw, x + w + pad_x)
                ay1 = max(0, y - 2)
                ay2 = min(vh, y + h + pad_below)
                if ax2 > ax1 and ay2 > ay1:
                    valid[ay1:ay2, ax1:ax2] = 0

    def blacklist_point(self, x: float, y: float, now: float, ttl_sec: float = 120.0) -> None:
        """Ignore an unreachable/blocked Gold Bag coordinate for ttl_sec (default 120s)."""
        self._blacklist = [b for b in self._blacklist if now < b[0]]
        self._blacklist.append((now + max(1.0, float(ttl_sec)), float(x), float(y)))

    def is_blacklisted(self, x: float, y: float, now: float, scale_x: float) -> bool:
        if not self._blacklist:
            return False
        self._blacklist = [b for b in self._blacklist if now < b[0]]
        r = 38.0 * max(0.55, scale_x)
        return any(math.hypot(x - bx, y - by) <= r for _exp, bx, by in self._blacklist)

    def _detect_dead_corpses(
        self,
        hsv_crop: np.ndarray,
        ox: int,
        oy: int,
        scale_x: float,
        scale_y: float,
    ) -> list[tuple[int, int, int, int]]:
        """Detect horizontal prone player corpses ('mayat') at 0.5x stride (<1ms CPU) to reject fake Gold Bags."""
        if hsv_crop.shape[0] < 16 or hsv_crop.shape[1] < 32:
            return []
        hsv_s = hsv_crop[::2, ::2]
        min_cue = max(6, int(round(11.0 * scale_x * scale_y)))

        # Fast C++ SIMD thresholding with immediate early-exits when any corpse clothing cue is absent
        _, dark = cv2.threshold(hsv_s[:, :, 2], 42, 255, cv2.THRESH_BINARY_INV)
        if cv2.countNonZero(dark) < min_cue:
            return []
        silver = cv2.inRange(hsv_s, self._silver_lo, self._silver_hi)
        if cv2.countNonZero(silver) < min_cue:
            return []
        tunic = cv2.inRange(hsv_s, self._tunic_lo, self._tunic_hi)
        if cv2.countNonZero(tunic) < min_cue:
            return []

        fg = cv2.bitwise_or(cv2.bitwise_or(dark, silver), tunic)
        kx = max(3, int(round(4.5 * scale_x)))
        ky = max(2, int(round(2.5 * scale_y)))
        kkey = (kx, ky)
        kern = self._corpse_kernels.get(kkey)
        if kern is None:
            kern = cv2.getStructuringElement(cv2.MORPH_RECT, kkey)
            self._corpse_kernels[kkey] = kern
        closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kern)
        n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        corpses: list[tuple[int, int, int, int]] = []
        min_w = max(16, int(round(26.0 * scale_x)))
        max_w = int(round(82.0 * scale_x))
        min_h = max(5, int(round(7.0 * scale_y)))
        max_h = int(round(29.0 * scale_y))

        for idx in range(1, n_lbl):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if not (min_w <= w <= max_w and min_h <= h <= max_h):
                continue
            if (w / max(1.0, float(h))) < 1.70:
                continue
            if (
                cv2.countNonZero(dark[y : y + h, x : x + w]) >= min_cue
                and cv2.countNonZero(silver[y : y + h, x : x + w]) >= min_cue
                and cv2.countNonZero(tunic[y : y + h, x : x + w]) >= min_cue
            ):
                corpses.append((ox + x * 2, oy + y * 2, ox + (x + w) * 2, oy + (y + h) * 2))
        return corpses

    def _check_pickup_chat(self, frame: FramePacket) -> bool:
        """Detect new 'Picked up' lines in the bottom-right chat log."""
        crop, _ = frame.crop_norm(self.chat_roi)
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 40:
            return False
        bottom = crop[int(crop.shape[0] * 0.65) :, :]
        sig = cv2.resize(bottom, (48, 16), interpolation=cv2.INTER_AREA).tobytes()
        changed = self._last_chat_sig is not None and sig != self._last_chat_sig
        self._last_chat_sig = sig
        if not changed or self.pickup_tmpl is None:
            return False
        th, tw = self.pickup_tmpl.shape[:2]
        if bottom.shape[0] < th or bottom.shape[1] < tw:
            return False
        res = cv2.matchTemplate(bottom, self.pickup_tmpl, cv2.TM_CCOEFF_NORMED)
        return float(cv2.minMaxLoc(res)[1]) >= 0.74

    def detect(
        self,
        frame: FramePacket,
        *,
        mobs: tuple[MobTarget, ...] = (),
        other_player_boxes: tuple[tuple[int, int, int, int], ...] = (),
        shared_playfield_hsv: np.ndarray | None = None,
        max_dist_override_px: float | None = None,
    ) -> tuple[tuple[LootDrop, ...], bool]:
        """Detect dropped Gold Bags in loot_area ROI and return `(drops, pickup_log_changed)`."""
        fw, fh = frame.width, frame.height
        self.last_corpse_boxes = ()
        if fw < 64 or fh < 64:
            return ((), False)

        pickup_changed = self._check_pickup_chat(frame)
        pf_rect = roi_to_pixels(self.playfield_roi, fw, fh)
        lx1, ly1, lx2, ly2 = roi_to_pixels(self.loot_roi, fw, fh)
        # Clamp loot ROI inside playfield
        ox = max(pf_rect[0], lx1)
        oy = max(pf_rect[1], ly1)
        ex = min(pf_rect[2], lx2)
        ey = min(pf_rect[3], ly2)
        if ex - ox < 24 or ey - oy < 24:
            return ((), pickup_changed)

        if shared_playfield_hsv is not None:
            sx0, sy0 = ox - pf_rect[0], oy - pf_rect[1]
            hsv = shared_playfield_hsv[sy0 : sy0 + (ey - oy), sx0 : sx0 + (ex - ox)]
        else:
            hsv = cv2.cvtColor(frame.image_bgr[oy:ey, ox:ex], cv2.COLOR_BGR2HSV)

        scale_x = fw / 1920.0
        scale_y = fh / 1080.0
        (feet_x, feet_y), _self_box = compute_self_zone(
            fw, fh, player_center_norm=self.player_center_norm, hud_fixed_pixel_ui=self.hud_fixed_pixel_ui
        )

        # Build exclusion mask (HUD + above horizon + self torso/armor + mobs + other players + prone dead corpses)
        hud_pf = build_playfield_hud_mask(fw, fh, pf_rect, hud_fixed_pixel_ui=self.hud_fixed_pixel_ui)
        sx0, sy0 = ox - pf_rect[0], oy - pf_rect[1]
        valid = hud_pf[sy0 : sy0 + (ey - oy), sx0 : sx0 + (ex - ox)].copy()

        # Horizon guard: reject sky/mountain/hill false positives above the ground horizon (y < 230 at 1024x768)
        horizon_y = (
            self.horizon_min_y_px
            if self.hud_fixed_pixel_ui
            else int(round(self.horizon_min_y_px * (fh / 768.0)))
        )
        hz_local = max(0, min(valid.shape[0], horizon_y - oy))
        if hz_local > 0:
            valid[:hz_local, :] = 0

        # Mask self character torso/armor (keeps ground directly below feet open for close drops!)
        self_px = int(round(fw * self.player_center_norm[0]))
        self_py = int(round(fh * self.player_center_norm[1]))
        sbx1 = max(0, int(round(self_px - 44.0 * scale_x)) - ox)
        sby1 = max(0, int(round(self_py - 95.0 * scale_y)) - oy)
        sbx2 = min(valid.shape[1], int(round(self_px + 46.0 * scale_x)) - ox)
        sby2 = min(valid.shape[0], int(round(self_py + 24.0 * scale_y)) - oy)
        if sbx2 > sbx1 and sby2 > sby1:
            valid[sby1:sby2, sbx1:sbx2] = 0

        pad_mx = int(round(34.0 * scale_x))
        pad_top = int(round(12.0 * scale_y))
        pad_bot = int(round(40.0 * scale_y))
        for m in mobs:
            mx1, my1, mx2, my2 = m.body_bbox
            nx, ny, nw, nh = m.nameplate_bbox
            for rx1, ry1, rx2, ry2 in (
                (mx1 - pad_mx, my1 - pad_top, mx2 + pad_mx, my2 + pad_bot),
                (nx - 8, ny - 4, nx + nw + 8, ny + nh + 18),
            ):
                ax1, ay1 = max(0, rx1 - ox), max(0, ry1 - oy)
                ax2, ay2 = min(valid.shape[1], rx2 - ox), min(valid.shape[0], ry2 - oy)
                if ax2 > ax1 and ay2 > ay1:
                    valid[ay1:ay2, ax1:ax2] = 0

        for pb in other_player_boxes:
            ax1, ay1 = max(0, pb[0] - pad_mx - ox), max(0, pb[1] - pad_top - oy)
            ax2, ay2 = min(valid.shape[1], pb[2] + pad_mx - ox), min(valid.shape[0], pb[3] + pad_bot - oy)
            if ax2 > ax1 and ay2 > ay1:
                valid[ay1:ay2, ax1:ax2] = 0

        corpses = self._detect_dead_corpses(hsv, ox, oy, scale_x, scale_y)
        self.last_corpse_boxes = tuple(corpses)
        for cx1, cy1, cx2, cy2 in corpses:
            ax1, ay1 = max(0, cx1 - 8 - ox), max(0, cy1 - 8 - oy)
            ax2, ay2 = min(valid.shape[1], cx2 + 8 - ox), min(valid.shape[0], cy2 + 8 - oy)
            if ax2 > ax1 and ay2 > ay1:
                valid[ay1:ay2, ax1:ax2] = 0

        # Guard: ignore yellow/gold armor gleam directly underneath any green or red HP bar strip (up to +110px)
        self._mask_below_hp_bar_strips(hsv, valid, scale_x, scale_y)

        v_ch = hsv[:, :, 2]
        raw_hits: list[tuple[int, int, int, int, float]] = []
        sur_pad_x = max(10, int(round(18.0 * scale_x)))
        sur_pad_y = max(10, int(round(18.0 * scale_y)))
        max_sur_warm = max(85, int(round(130.0 * scale_x * scale_y)))
        max_in_warm = max(28, int(round(46.0 * scale_x * scale_y)))

        def _is_isolated_bag(lx: int, ly: int, lw: int, lh: int) -> bool:
            """Reject metallic monster/armor patches that have large surrounding golden-warm regions."""
            in_patch = hsv[ly : ly + lh, lx : lx + lw]
            if cv2.countNonZero(cv2.inRange(in_patch, self._warm_lo, self._warm_hi)) > max_in_warm:
                return False
            sx1 = max(0, lx - sur_pad_x)
            sy1 = max(0, ly - sur_pad_y)
            sx2 = min(hsv.shape[1], lx + lw + sur_pad_x)
            sy2 = min(hsv.shape[0], ly + lh + sur_pad_y)
            sur_patch = hsv[sy1:sy2, sx1:sx2]
            return cv2.countNonZero(cv2.inRange(sur_patch, self._sur_warm_lo, self._sur_warm_hi)) <= max_sur_warm

        # Cue A: Bright orange-gold sparkle sitting directly above/inside a compact dark pouch core
        glow = cv2.bitwise_and(cv2.inRange(hsv, self.glow_lo, self.glow_hi), valid)
        min_b = max(2, int(round(self.min_blob_px * scale_x * scale_y)))
        max_b = max(35, int(round(self.max_blob_px * scale_x * scale_y)))
        if cv2.countNonZero(glow) >= min_b:
            n_g, _lg, st_g, _ = cv2.connectedComponentsWithStats(glow, connectivity=8)
            for idx in range(1, n_g):
                area = int(st_g[idx, cv2.CC_STAT_AREA])
                if not (min_b <= area <= max_b):
                    continue
                x = int(st_g[idx, cv2.CC_STAT_LEFT])
                y = int(st_g[idx, cv2.CC_STAT_TOP])
                w = int(st_g[idx, cv2.CC_STAT_WIDTH])
                h = int(st_g[idx, cv2.CC_STAT_HEIGHT])
                if w > int(round(28 * scale_x)) or h > int(round(24 * scale_y)):
                    continue
                ry1 = min(hsv.shape[0] - 1, y + h)
                ry2 = min(hsv.shape[0], ry1 + max(6, int(round(15.0 * scale_y))))
                rx1 = max(0, x - max(3, int(round(6.0 * scale_x))))
                rx2 = min(hsv.shape[1], x + w + max(3, int(round(6.0 * scale_x))))
                if ry2 <= ry1 or rx2 <= rx1:
                    continue
                patch_v = v_ch[ry1:ry2, rx1:rx2]
                dark_r = float((patch_v <= self.dark_body_v_max).mean())
                if dark_r >= self.min_dark_ratio:
                    bw = max(10, rx2 - rx1)
                    bh = max(12, ry2 - y)
                    if _is_isolated_bag(rx1, y, bw, bh):
                        raw_hits.append((ox + rx1, oy + y, bw, bh, min(1.0, 0.65 + dark_r * 0.35)))

        # Cue B: Compact dark pouch with warm red-orange ribbon (dimmed / dusk bags)
        med_v = float(np.median(v_ch[::8, ::8])) if v_ch.size > 0 else 65.0
        eff_pouch_v = min(self.pouch_v_max, max(24, int(med_v * 0.68)))
        _, dark_raw = cv2.threshold(v_ch, eff_pouch_v, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_raw, valid)
        min_pa = max(18, int(round(self.pouch_min_area * scale_x * scale_y)))
        max_pa = max(160, int(round(self.pouch_max_area * scale_x * scale_y)))
        min_warm = max(4, int(round(self.pouch_warm_min_px * scale_x * scale_y)))

        if cv2.countNonZero(dark_mask) >= min_pa:
            warm_ribbon = cv2.inRange(hsv, self._warm_lo, self._warm_hi)
            n_d, _ld, st_d, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
            max_pw = int(round(28 * scale_x))
            max_ph = int(round(28 * scale_y))
            for idx in range(1, n_d):
                area = int(st_d[idx, cv2.CC_STAT_AREA])
                if not (min_pa <= area <= max_pa):
                    continue
                x = int(st_d[idx, cv2.CC_STAT_LEFT])
                y = int(st_d[idx, cv2.CC_STAT_TOP])
                w = int(st_d[idx, cv2.CC_STAT_WIDTH])
                h = int(st_d[idx, cv2.CC_STAT_HEIGHT])
                if w > max_pw or h > max_ph:
                    continue
                asp = w / max(1.0, float(h))
                if not (0.45 <= asp <= 2.1):
                    continue
                ry1 = max(0, y - max(4, int(round(7.0 * scale_y))))
                ry2 = min(hsv.shape[0], y + h + 2)
                rx1 = max(0, x - 3)
                rx2 = min(hsv.shape[1], x + w + 3)
                w_cnt = cv2.countNonZero(warm_ribbon[ry1:ry2, rx1:rx2])
                if w_cnt >= min_warm and _is_isolated_bag(x, ry1, w, ry2 - ry1):
                    raw_hits.append((ox + x, oy + ry1, w, ry2 - ry1, min(0.95, 0.55 + w_cnt / 40.0)))

        # Cluster adjacent hits within cluster_radius_px
        cl_r = self.cluster_radius_px * max(0.55, scale_x)
        clusters: list[tuple[int, int, int, int, float]] = []
        for bx, by, bw, bh, sc in sorted(raw_hits, key=lambda item: -item[4]):
            cx = bx + bw // 2
            cy = by + bh // 2
            if any(math.hypot(cx - (ex0 + ew0 // 2), cy - (ey0 + eh0 // 2)) <= cl_r for ex0, ey0, ew0, eh0, _ in clusters):
                continue
            clusters.append((bx, by, bw, bh, sc))

        if max_dist_override_px is not None:
            max_d = float(max_dist_override_px) * (1.0 if self.hud_fixed_pixel_ui else max(0.55, scale_x))
        else:
            max_d = self.max_dist_px * max(0.55, scale_x)
        drops: list[LootDrop] = []
        for bx, by, bw, bh, sc in clusters:
            cx = bx + bw // 2
            cy = by + bh // 2
            if self.is_blacklisted(cx, cy, frame.timestamp, scale_x):
                continue
            dist = math.hypot(cx - feet_x, cy - feet_y)
            if dist > max_d:
                continue
            drops.append(
                LootDrop(
                    bbox=(bx, by, bw, bh),
                    click_x=cx,
                    click_y=cy,
                    dist_to_player_px=float(dist),
                    score=round(sc, 3),
                )
            )

        drops.sort(key=lambda d: d.dist_to_player_px)
        return (tuple(drops), pickup_changed)
