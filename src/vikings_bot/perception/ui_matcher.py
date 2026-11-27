"""UI template matcher for town Addon Keep Teleport NPC sign and Talk/Teleport popup menu."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig

logger = logging.getLogger(__name__)


class UiTemplateMatcher:
    """Finds Addon Keep Teleport NPC sign and its Talk/Teleport popup menu via normalized cross-correlation."""

    def __init__(self, config: BotConfig) -> None:
        rt = config.respawn_travel
        rep = config.repair
        self.sign_thresh = float(rt.get("sign_match_threshold", 0.66))
        self.menu_thresh = float(rt.get("menu_match_threshold", 0.78))
        self.ok_thresh = float(rt.get("npc_talk_ok_threshold", 0.72))
        self.merchant_sign_thresh = float(rep.get("merchant_sign_threshold", 0.66))
        self.trade_menu_thresh = float(rep.get("trade_menu_threshold", 0.78))
        self.repair_btn_thresh = float(rep.get("repair_btn_threshold", 0.72))
        self.repair_yes_thresh = float(rep.get("repair_yes_threshold", 0.72))
        self.no_equip_repair_thresh = float(rep.get("no_equipment_repair_threshold", 0.70))
        self.inventory_tab_thresh = float(rep.get("inventory_tab_threshold", 0.74))
        self.hud_fixed_pixel_ui = bool(config.perception.get("hud_fixed_pixel_ui", False))
        p_roi = config.perception.get("interface_mode_roi", [0.12, 0.05, 0.88, 0.22])
        self.interface_mode_roi: tuple[float, float, float, float] = (
            float(p_roi[0]),
            float(p_roi[1]),
            float(p_roi[2]),
            float(p_roi[3]),
        )
        self.interface_mode_thresh = float(config.perception.get("interface_mode_threshold", 0.68))
        self._sticky_control_mode: str | None = None
        self._last_control_mode_source: str = "uninitialized"

        sign_path = config.resolve_path(
            str(rt.get("addon_keep_sign_template", "assets/ui_templates/addon_keep_sign.png"))
        )
        menu_path = config.resolve_path(
            str(rt.get("teleport_menu_template", "assets/ui_templates/teleport_menu.png"))
        )
        ok_path = config.resolve_path(
            str(rt.get("npc_talk_ok_template", "assets/ui_templates/npc_talk_ok.png"))
        )
        merchant_path = config.resolve_path(
            str(rep.get("merchant_sign_template", "assets/ui_templates/merchant_sign.png"))
        )
        trade_menu_path = config.resolve_path(
            str(rep.get("trade_menu_template", "assets/ui_templates/trade_menu.png"))
        )
        repair_all_path = config.resolve_path(
            str(rep.get("repair_all_template", "assets/ui_templates/repair_all_btn.png"))
        )
        repair_yes_path = config.resolve_path(
            str(rep.get("repair_yes_template", "assets/ui_templates/repair_yes_btn.png"))
        )
        no_equip_path = config.resolve_path(
            str(rep.get("no_equipment_repair_template", "assets/ui_templates/no_equipment_repair.png"))
        )
        inv_tab_path = config.resolve_path(
            str(rep.get("inventory_tab_template", "assets/ui_templates/inventory_tab.png"))
        )
        kb_banner_path = config.resolve_path("assets/ui_templates/interface_keyboard_banner.png")
        ms_banner_path = config.resolve_path("assets/ui_templates/interface_mouse_banner.png")
        self.sign_tmpl: np.ndarray | None = cv2.imread(str(sign_path), cv2.IMREAD_COLOR) if sign_path.exists() else None
        self.menu_tmpl: np.ndarray | None = cv2.imread(str(menu_path), cv2.IMREAD_COLOR) if menu_path.exists() else None
        self.ok_tmpl: np.ndarray | None = cv2.imread(str(ok_path), cv2.IMREAD_COLOR) if ok_path.exists() else None
        self.merchant_sign_tmpl: np.ndarray | None = (
            cv2.imread(str(merchant_path), cv2.IMREAD_COLOR) if merchant_path.exists() else None
        )
        self.trade_menu_tmpl: np.ndarray | None = (
            cv2.imread(str(trade_menu_path), cv2.IMREAD_COLOR) if trade_menu_path.exists() else None
        )
        self.repair_all_tmpl: np.ndarray | None = (
            cv2.imread(str(repair_all_path), cv2.IMREAD_COLOR) if repair_all_path.exists() else None
        )
        self.repair_yes_tmpl: np.ndarray | None = (
            cv2.imread(str(repair_yes_path), cv2.IMREAD_COLOR) if repair_yes_path.exists() else None
        )
        self.no_equip_repair_tmpl: np.ndarray | None = (
            cv2.imread(str(no_equip_path), cv2.IMREAD_COLOR) if no_equip_path.exists() else None
        )
        if self.no_equip_repair_tmpl is None:
            logger.warning("Optional UI template 'no_equipment_repair' not found at %s (will rely on OK/timeout)", no_equip_path)
        self.inv_tab_tmpl: np.ndarray | None = (
            cv2.imread(str(inv_tab_path), cv2.IMREAD_COLOR) if inv_tab_path.exists() else None
        )
        self.kb_banner_tmpl: np.ndarray | None = (
            cv2.imread(str(kb_banner_path), cv2.IMREAD_COLOR) if kb_banner_path.exists() else None
        )
        self.ms_banner_tmpl: np.ndarray | None = (
            cv2.imread(str(ms_banner_path), cv2.IMREAD_COLOR) if ms_banner_path.exists() else None
        )
        self._sign_g2: np.ndarray | None = (
            cv2.resize(cv2.cvtColor(self.sign_tmpl, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if self.sign_tmpl is not None
            else None
        )
        self._menu_g2: np.ndarray | None = (
            cv2.resize(cv2.cvtColor(self.menu_tmpl, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if self.menu_tmpl is not None
            else None
        )
        self._merchant_sign_g2: np.ndarray | None = (
            cv2.resize(
                cv2.cvtColor(self.merchant_sign_tmpl, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )
            if self.merchant_sign_tmpl is not None
            else None
        )
        self._trade_menu_g2: np.ndarray | None = (
            cv2.resize(
                cv2.cvtColor(self.trade_menu_tmpl, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )
            if self.trade_menu_tmpl is not None
            else None
        )
        self._banner_cache: dict[tuple[int, int, bool], tuple[np.ndarray, np.ndarray]] = {}
        self.last_match_scores: dict[str, float] = {}
        self._last_miss_log_ts: dict[str, float] = {}

        if self.sign_tmpl is None:
            logger.debug("Addon Keep sign template not found at %s", sign_path)
        if self.menu_tmpl is None:
            logger.debug("Teleport menu template not found at %s", menu_path)
        if self.ok_tmpl is None:
            logger.debug("NPC Talk [OK] template not found at %s", ok_path)

    @staticmethod
    def _scale_template(tmpl: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
        sx = frame_w / 1920.0
        sy = frame_h / 1080.0
        if abs(sx - 1.0) <= 0.03 and abs(sy - 1.0) <= 0.03:
            return tmpl
        nw = max(8, int(round(tmpl.shape[1] * sx)))
        nh = max(6, int(round(tmpl.shape[0] * sy)))
        return cv2.resize(tmpl, (nw, nh), interpolation=cv2.INTER_AREA)

    def _record_template_score(
        self,
        name: str,
        score: float,
        threshold: float,
        matched: bool,
        ts: float,
    ) -> None:
        if not name:
            return
        self.last_match_scores[name] = float(score)
        if not matched:
            prev = self._last_miss_log_ts.get(name, -999.0)
            if (ts - prev) >= 2.0:
                self._last_miss_log_ts[name] = ts
                logger.info(
                    "UI template '%s' miss: best_score=%.3f < thresh=%.3f",
                    name,
                    float(score),
                    float(threshold),
                )

    def _match_in_roi(
        self,
        frame: FramePacket,
        tmpl: np.ndarray | None,
        tmpl_g2: np.ndarray | None,
        roi_norm: tuple[float, float, float, float],
        threshold: float,
        *,
        template_name: str = "",
    ) -> tuple[int, int, int, int] | None:
        if tmpl is None or frame.image_bgr.size == 0:
            return None
        crop, (ox, oy, _x2, _y2) = frame.crop_norm(roi_norm)
        best_seen = -1.0

        # In RYL / Vikings Online at 1024x768, UI elements & NPC sign frames render at native 1:1 pixels
        # (not scaled by 1024/1920 = 0.53x), so try native 1:1 matching first when hud_fixed_pixel_ui=True!
        if self.hud_fixed_pixel_ui:
            th_n, tw_n = tmpl.shape[:2]
            if crop.shape[0] >= th_n and crop.shape[1] >= tw_n:
                res_n = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
                _min_n, max_n, _min_ln, max_ln = cv2.minMaxLoc(res_n)
                best_seen = max(best_seen, float(max_n))
                if float(max_n) >= threshold:
                    self._record_template_score(template_name, float(max_n), threshold, True, frame.timestamp)
                    return (ox + int(max_ln[0]), oy + int(max_ln[1]), int(tw_n), int(th_n))

        scaled = self._scale_template(tmpl, frame.width, frame.height)
        th, tw = scaled.shape[:2]
        if crop.shape[0] < th or crop.shape[1] < tw:
            if best_seen >= 0.0:
                self._record_template_score(template_name, best_seen, threshold, False, frame.timestamp)
            return None

        # Fast 2-stage coarse-to-fine matching (~6x faster at 1080p with bit-exact 1x BGR verification!)
        if frame.width >= 1400 and tmpl_g2 is not None and abs(frame.width / 1920.0 - 1.0) <= 0.03:
            g_half = cv2.cvtColor(
                cv2.resize(crop, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
            )
            if g_half.shape[0] >= tmpl_g2.shape[0] and g_half.shape[1] >= tmpl_g2.shape[1]:
                res_c = cv2.matchTemplate(g_half, tmpl_g2, cv2.TM_CCOEFF_NORMED)
                _min_c, max_c, _min_lc, max_lc = cv2.minMaxLoc(res_c)
                best_seen = max(best_seen, float(max_c))
                if float(max_c) < max(0.35, threshold - 0.25):
                    self._record_template_score(template_name, best_seen, threshold, False, frame.timestamp)
                    return None
                cx0, cy0 = int(max_lc[0]) * 2, int(max_lc[1]) * 2
                pad = 16
                x1 = max(0, cx0 - pad)
                y1 = max(0, cy0 - pad)
                x2 = min(crop.shape[1], cx0 + tw + pad)
                y2 = min(crop.shape[0], cy0 + th + pad)
                patch = crop[y1:y2, x1:x2]
                if patch.shape[0] >= th and patch.shape[1] >= tw:
                    res_f = cv2.matchTemplate(patch, scaled, cv2.TM_CCOEFF_NORMED)
                    _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res_f)
                    best_seen = max(best_seen, float(max_v))
                    if float(max_v) < threshold:
                        self._record_template_score(template_name, best_seen, threshold, False, frame.timestamp)
                        return None
                    self._record_template_score(template_name, float(max_v), threshold, True, frame.timestamp)
                    return (ox + x1 + int(max_l[0]), oy + y1 + int(max_l[1]), int(tw), int(th))

        res = cv2.matchTemplate(crop, scaled, cv2.TM_CCOEFF_NORMED)
        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
        best_seen = max(best_seen, float(max_v))
        if float(max_v) < threshold:
            self._record_template_score(template_name, best_seen, threshold, False, frame.timestamp)
            return None
        self._record_template_score(template_name, float(max_v), threshold, True, frame.timestamp)
        return (ox + int(max_l[0]), oy + int(max_l[1]), int(tw), int(th))

    @staticmethod
    def _build_sign_fg_mask(tmpl_bgr: np.ndarray) -> np.ndarray:
        """Mask out semi-transparent interior background; keep golden frame + bright NPC name text."""
        hsv = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2HSV)
        gold_frame = (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 75)
        bright_text = hsv[:, :, 2] >= 85
        mask = ((gold_frame | bright_text).astype(np.uint8)) * 255
        return cv2.merge([mask, mask, mask])

    def _match_sign_in_roi(
        self,
        frame: FramePacket,
        tmpl: np.ndarray | None,
        tmpl_g2: np.ndarray | None,
        roi_norm: tuple[float, float, float, float],
        threshold: float,
        *,
        template_name: str,
    ) -> tuple[int, int, int, int] | None:
        hit = self._match_in_roi(
            frame,
            tmpl,
            tmpl_g2,
            roi_norm=roi_norm,
            threshold=threshold,
            template_name=template_name,
        )
        if hit is not None or tmpl is None or frame.image_bgr.size == 0:
            return hit

        # Semi-transparent NPC sign fallback: match ONLY golden border + text pixels (ignores varying grass/wall background)
        crop, (ox, oy, _x2, _y2) = frame.crop_norm(roi_norm)
        t_use = tmpl if self.hud_fixed_pixel_ui else self._scale_template(tmpl, frame.width, frame.height)
        th, tw = t_use.shape[:2]
        if crop.shape[0] < th or crop.shape[1] < tw:
            return None
        fg_mask = self._build_sign_fg_mask(t_use)
        if int((fg_mask[:, :, 0] > 0).sum()) < 24:
            return None
        res_m = cv2.matchTemplate(crop, t_use, cv2.TM_CCOEFF_NORMED, mask=fg_mask)
        res_m = np.nan_to_num(res_m, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _min_m, max_m, _min_lm, max_lm = cv2.minMaxLoc(res_m)
        best_m = float(max_m)
        if best_m >= threshold:
            self._record_template_score(template_name, best_m, threshold, True, frame.timestamp)
            return (ox + int(max_lm[0]), oy + int(max_lm[1]), int(tw), int(th))
        prev_best = self.last_match_scores.get(template_name, -1.0)
        if best_m > prev_best:
            self.last_match_scores[template_name] = best_m
        return None

    def _compute_popup_roi_norm(
        self,
        frame: FramePacket,
        default_roi: tuple[float, float, float, float],
        anchor_xy: tuple[int, int] | None,
    ) -> tuple[float, float, float, float]:
        """Forbid detections in top sky strip (y < 140 on 1024x768, scaled) and constrain around last NPC click when known."""
        fw = max(1, frame.width)
        fh = max(1, frame.height)
        sx = 1.0 if self.hud_fixed_pixel_ui else (fw / 1024.0)
        sy = 1.0 if self.hud_fixed_pixel_ui else (fh / 768.0)
        min_y_px = int(round(140.0 * sy))
        if anchor_xy is None:
            x1n, y1n, x2n, y2n = default_roi
            return (x1n, max(y1n, min_y_px / fh), x2n, y2n)
        ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
        x1 = max(0, int(round(ax - 180.0 * sx)))
        x2 = min(fw, int(round(ax + 180.0 * sx)))
        y1 = max(min_y_px, int(round(ay - 240.0 * sy)))
        y2 = min(fh, int(round(ay + 200.0 * sy)))
        if (x2 - x1) < 40 or (y2 - y1) < 30:
            x1n, y1n, x2n, y2n = default_roi
            return (x1n, max(y1n, min_y_px / fh), x2n, y2n)
        return (x1 / fw, y1 / fh, x2 / fw, y2 / fh)

    def find_addon_keep_sign(self, frame: FramePacket) -> tuple[int, int, int, int] | None:
        """Locate the golden-framed 'Addon Keep Teleport' NPC nameplate in the playfield."""
        return self._match_sign_in_roi(
            frame,
            self.sign_tmpl,
            self._sign_g2,
            roi_norm=(0.06, 0.08, 0.94, 0.78),
            threshold=self.sign_thresh,
            template_name="addon_keep_sign",
        )

    def find_teleport_menu(
        self,
        frame: FramePacket,
        *,
        anchor_xy: tuple[int, int] | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Locate the 'Talk / Teleport' popup box near the clicked Addon Keep Teleport NPC (forbids sky y<140)."""
        roi = self._compute_popup_roi_norm(frame, (0.10, 0.185, 0.92, 0.78), anchor_xy)
        return self._match_in_roi(
            frame,
            self.menu_tmpl,
            self._menu_g2,
            roi_norm=roi,
            threshold=self.menu_thresh,
            template_name="teleport_menu",
        )

    def find_merchant_sign(self, frame: FramePacket) -> tuple[int, int, int, int] | None:
        """Locate the golden-framed 'Beginner Armor Merchant' NPC nameplate in town."""
        return self._match_sign_in_roi(
            frame,
            self.merchant_sign_tmpl,
            self._merchant_sign_g2,
            roi_norm=(0.06, 0.08, 0.94, 0.80),
            threshold=self.merchant_sign_thresh,
            template_name="merchant_sign",
        )

    def find_trade_menu(
        self,
        frame: FramePacket,
        *,
        anchor_xy: tuple[int, int] | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Locate the 'Talk / Trade' popup box near the clicked Beginner Armor Merchant NPC (forbids sky y<140)."""
        roi = self._compute_popup_roi_norm(frame, (0.10, 0.185, 0.92, 0.78), anchor_xy)
        return self._match_in_roi(
            frame,
            self.trade_menu_tmpl,
            self._trade_menu_g2,
            roi_norm=roi,
            threshold=self.trade_menu_thresh,
            template_name="trade_menu",
        )

    def find_repair_all_btn(self, frame: FramePacket) -> tuple[int, int, int, int] | None:
        """Locate the 'Repair All' button inside the open Merchant Shop window (1080p & 1024x768)."""
        return self._match_in_roi(
            frame,
            self.repair_all_tmpl,
            None,
            roi_norm=(0.30, 0.32, 0.98, 0.88),
            threshold=self.repair_btn_thresh,
            template_name="repair_all",
        )

    def find_repair_yes_btn(self, frame: FramePacket) -> tuple[int, int, int, int] | None:
        """Locate the modal '[ Yes ]' confirmation button ('It costs ... gold to repair the item')."""
        roi_yes = (0.32, 0.36, 0.74, 0.74) if self.hud_fixed_pixel_ui else (0.40, 0.40, 0.65, 0.66)
        hit = self._match_in_roi(
            frame,
            self.repair_yes_tmpl,
            None,
            roi_norm=roi_yes,
            threshold=self.repair_yes_thresh,
            template_name="repair_yes",
        )
        if hit is None:
            return None
        fx, fy, tw, th = hit
        sx = 1.0 if self.hud_fixed_pixel_ui else (frame.width / 1920.0)
        sy = 1.0 if self.hud_fixed_pixel_ui else (frame.height / 1080.0)
        bx1 = max(0, int(round(fx - 110.0 * sx)))
        bx2 = max(bx1 + 4, int(round(fx + 20.0 * sx)))
        by1 = max(0, int(round(fy - 55.0 * sy)))
        by2 = max(by1 + 4, int(round(fy - 4.0 * sy)))
        interior = frame.image_bgr[by1:by2, bx1:bx2]
        if interior.size == 0:
            return None
        v_chan = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)[:, :, 2]
        if float((v_chan < 85).mean()) < 0.55:
            return None
        return (fx, fy, tw, th)

    def find_no_equipment_repair(self, frame: FramePacket) -> tuple[int, int, int, int] | None:
        """Locate the 'No equipment to repair.' message/dialog after clicking 'Repair All'."""
        if self.no_equip_repair_tmpl is None:
            return None
        return self._match_in_roi(
            frame,
            self.no_equip_repair_tmpl,
            None,
            roi_norm=(0.20, 0.25, 0.82, 0.78),
            threshold=self.no_equip_repair_thresh,
            template_name="no_equipment_repair",
        )

    def find_inventory_tab(self, frame: FramePacket) -> tuple[int, int, int, int] | None:
        """Locate the 'Inventory | Skill' header tab of the open Inventory window."""
        return self._match_in_roi(
            frame,
            self.inv_tab_tmpl,
            None,
            roi_norm=(0.45, 0.04, 1.00, 0.45),
            threshold=self.inventory_tab_thresh,
            template_name="inventory_tab",
        )

    def find_npc_talk_dialog(self, frame: FramePacket) -> tuple[int, int, int, int] | None:
        """Locate the modal NPC Talk dialog '[ OK ]' button (e.g. accidental click on 'Talk' instead of 'Teleport').

        Matches the golden-framed [ OK ] button in the center screen ROI AND verifies the dark NPC
        dialog interior immediately up-left of the button (100% recall, 0% false positives, <0.4ms).
        """
        if self.ok_tmpl is None or frame.image_bgr.size == 0:
            return None
        roi_ok = (0.34, 0.36, 0.76, 0.74) if self.hud_fixed_pixel_ui else (0.44, 0.43, 0.63, 0.63)
        crop, (ox, oy, _x2, _y2) = frame.crop_norm(roi_ok)

        matched_native = False
        if self.hud_fixed_pixel_ui and crop.shape[0] >= self.ok_tmpl.shape[0] and crop.shape[1] >= self.ok_tmpl.shape[1]:
            res_n = cv2.matchTemplate(crop, self.ok_tmpl, cv2.TM_CCOEFF_NORMED)
            _min_n, max_n, _min_ln, max_ln = cv2.minMaxLoc(res_n)
            if float(max_n) >= self.ok_thresh:
                tw, th = int(self.ok_tmpl.shape[1]), int(self.ok_tmpl.shape[0])
                max_l = max_ln
                matched_native = True

        if not matched_native:
            scaled = self._scale_template(self.ok_tmpl, frame.width, frame.height)
            th, tw = scaled.shape[:2]
            if crop.shape[0] < th or crop.shape[1] < tw:
                return None

            res = cv2.matchTemplate(crop, scaled, cv2.TM_CCOEFF_NORMED)
            _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
            if float(max_v) < self.ok_thresh:
                return None

        fx = ox + int(max_l[0])
        fy = oy + int(max_l[1])
        sx = 1.0 if matched_native else (frame.width / 1920.0)
        sy = 1.0 if matched_native else (frame.height / 1080.0)
        bx1 = max(0, int(round(fx - 175.0 * sx)))
        bx2 = max(bx1 + 4, int(round(fx - 10.0 * sx)))
        by1 = max(0, int(round(fy - 85.0 * sy)))
        by2 = min(frame.height, max(by1 + 4, int(round(fy + 10.0 * sy))))
        interior = frame.image_bgr[by1:by2, bx1:bx2]
        if interior.size == 0:
            return None
        v_chan = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)[:, :, 2]
        dark_frac = float((v_chan < 80).mean())
        if dark_frac < 0.58:
            return None
        return (fx, fy, int(tw), int(th))

    @staticmethod
    def _has_white_gauntlet_in_bgr(patch_bgr: np.ndarray) -> bool:
        """Detect RYL's crisp white-with-dark-outline Mouse Mode gauntlet cursor inside a BGR patch."""
        if patch_bgr.size == 0 or patch_bgr.shape[0] < 10 or patch_bgr.shape[1] < 10:
            return False
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        wm = ((hsv[:, :, 1] <= 40) & (hsv[:, :, 2] >= 195)).astype(np.uint8) * 255
        n, _lbl, st, _ct = cv2.connectedComponentsWithStats(wm, connectivity=8)
        sh, sw = patch_bgr.shape[:2]
        for i in range(1, n):
            x = int(st[i, cv2.CC_STAT_LEFT])
            y = int(st[i, cv2.CC_STAT_TOP])
            w = int(st[i, cv2.CC_STAT_WIDTH])
            h = int(st[i, cv2.CC_STAT_HEIGHT])
            a = int(st[i, cv2.CC_STAT_AREA])
            if not (14 <= a <= 360 and 6 <= w <= 38 and 6 <= h <= 38):
                continue
            asp = w / max(1.0, float(h))
            if not (0.45 <= asp <= 1.90):
                continue
            y1, y2 = max(0, y - 3), min(sh, y + h + 3)
            x1, x2 = max(0, x - 3), min(sw, x + w + 3)
            ring_v = hsv[y1:y2, x1:x2, 2]
            if int((ring_v <= 75).sum()) >= 12:
                return True
        return False

    def _get_banner_templates(
        self, is_kb: bool, frame_w: int, frame_h: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return cached `(bgr_scaled, gray_half)` templates for the INTERFACE TYPE top banner."""
        key = (int(frame_w), int(frame_h), bool(is_kb))
        cached = self._banner_cache.get(key)
        if cached is not None:
            return cached
        tmpl = self.kb_banner_tmpl if is_kb else self.ms_banner_tmpl
        if tmpl is None:
            return None
        t_use = tmpl
        if (not self.hud_fixed_pixel_ui) and (abs(frame_w / 1024.0 - 1.0) > 0.05):
            sx = frame_w / 1024.0
            sy = frame_h / 768.0
            t_use = cv2.resize(
                tmpl,
                (max(16, int(round(tmpl.shape[1] * sx))), max(8, int(round(tmpl.shape[0] * sy)))),
                interpolation=cv2.INTER_AREA,
            )
        t_g2 = cv2.resize(
            cv2.cvtColor(t_use, cv2.COLOR_BGR2GRAY),
            (max(8, t_use.shape[1] // 2), max(4, t_use.shape[0] // 2)),
            interpolation=cv2.INTER_AREA,
        )
        if len(self._banner_cache) > 16:
            self._banner_cache.clear()
        self._banner_cache[key] = (t_use, t_g2)
        return (t_use, t_g2)

    def detect_interface_mode(
        self,
        frame: FramePacket,
        cursor_rel_xy: tuple[int, int] | None = None,
    ) -> tuple[str, str]:
        """Detect Vikings Online interface mode with zero false Keyboard-Mode triggers (<1ms).

        Returns (mode, reason) where mode is one of:
        - 'keyboard_banner': explicit 'INTERFACE TYPE: Keyboard' top banner (100% certain Keyboard Mode)
        - 'mouse_banner':    explicit 'INTERFACE TYPE: Mouse' top banner (100% certain Mouse Mode)
        - 'mouse_cursor':    white RYL gauntlet cursor visible (100% certain Mouse Mode)
        - 'no_cursor':       neither banner nor white gauntlet seen (ambiguous: could be Keyboard Mode
                             OR Mouse Mode with red attack-sword/hand/hardware cursor!)
        """
        if frame.image_bgr.size == 0:
            return ("unknown", "empty_frame")

        # 1) Top-center/sliding INTERFACE TYPE banner (positive-only, zero false positives).
        #    Uses fast 0.5x grayscale coarse pass (~0.5ms) + exact 1x BGR fine verification on candidates.
        if self.kb_banner_tmpl is not None or self.ms_banner_tmpl is not None:
            bcrop, _ = frame.crop_norm(self.interface_mode_roi)
            kb_score = 0.0
            ms_score = 0.0
            bh, bw = bcrop.shape[:2]
            bcrop_g2: np.ndarray | None = None
            if bh >= 16 and bw >= 32:
                bcrop_g2 = cv2.cvtColor(
                    cv2.resize(bcrop, (bw // 2, bh // 2), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )
            coarse_gate = max(0.36, self.interface_mode_thresh - 0.28)
            for is_kb in (True, False):
                pair = self._get_banner_templates(is_kb, frame.width, frame.height)
                if pair is None:
                    continue
                t_use, t_g2 = pair
                th, tw = t_use.shape[:2]
                if bh < th or bw < tw:
                    continue
                sc = 0.0
                if bcrop_g2 is not None and bcrop_g2.shape[0] >= t_g2.shape[0] and bcrop_g2.shape[1] >= t_g2.shape[1]:
                    res_c = cv2.matchTemplate(bcrop_g2, t_g2, cv2.TM_CCOEFF_NORMED)
                    _min_c, max_c, _min_lc, max_lc = cv2.minMaxLoc(res_c)
                    sc = float(max_c)
                    if sc >= coarse_gate:
                        cx0, cy0 = int(max_lc[0]) * 2, int(max_lc[1]) * 2
                        pad = 18
                        x1 = max(0, cx0 - pad)
                        y1 = max(0, cy0 - pad)
                        x2 = min(bw, cx0 + tw + pad)
                        y2 = min(bh, cy0 + th + pad)
                        patch = bcrop[y1:y2, x1:x2]
                        if patch.shape[0] >= th and patch.shape[1] >= tw:
                            sc = float(cv2.minMaxLoc(cv2.matchTemplate(patch, t_use, cv2.TM_CCOEFF_NORMED))[1])
                else:
                    sc = float(cv2.minMaxLoc(cv2.matchTemplate(bcrop, t_use, cv2.TM_CCOEFF_NORMED))[1])
                if is_kb:
                    kb_score = sc
                else:
                    ms_score = sc
            if kb_score >= self.interface_mode_thresh and kb_score > (ms_score + 0.10):
                return ("keyboard_banner", f"banner_keyboard({kb_score:.2f})")
            if ms_score >= self.interface_mode_thresh and ms_score > (kb_score + 0.10):
                return ("mouse_banner", f"banner_mouse({ms_score:.2f})")

        # 2) Tight 64x64 patch around live cursor position inside game client.
        #    Ignore exact screen center (fw/2, fh/2) +-14px because RYL locks the hidden cursor to center
        #    in Keyboard Mode right over the character's white HP bar; NEVER run a whole-playfield scan
        #    (which falsely matched white player nameplates like Kim-NiNi while still in Keyboard Mode!).
        if cursor_rel_xy is not None:
            cx, cy = int(cursor_rel_xy[0]), int(cursor_rel_xy[1])
            in_kb_center_lock = (abs(cx - frame.width // 2) <= 14) and (abs(cy - frame.height // 2) <= 14)
            if (not in_kb_center_lock) and 0 <= cx < frame.width and 0 <= cy < frame.height:
                x1, y1 = max(0, cx - 32), max(0, cy - 32)
                x2, y2 = min(frame.width, cx + 32), min(frame.height, cy + 32)
                if self._has_white_gauntlet_in_bgr(frame.image_bgr[y1:y2, x1:x2]):
                    return ("mouse_cursor", "gauntlet_at_cursor")

        return ("no_cursor", "no_gauntlet_visible")

    def detect_control_mode(
        self,
        frame: FramePacket,
        cursor_rel_xy: tuple[int, int] | None = None,
        manual_fallback: str = "keyboard",
    ) -> tuple[str, str, str]:
        """Auto-detect active RYL control mode ('mouse' vs 'keyboard') from top-HUD ESC banner + cursor.

        Returns (resolved_mode, source_reason, raw_interface_mode).
        """
        fallback = "mouse" if str(manual_fallback).lower() == "mouse" else "keyboard"
        raw_mode, raw_reason = self.detect_interface_mode(frame, cursor_rel_xy=cursor_rel_xy)
        if raw_mode == "keyboard_banner":
            self._sticky_control_mode = "keyboard"
            self._last_control_mode_source = f"top_hud:{raw_reason}"
            return ("keyboard", self._last_control_mode_source, raw_mode)
        if raw_mode in ("mouse_banner", "mouse_cursor"):
            self._sticky_control_mode = "mouse"
            self._last_control_mode_source = f"top_hud:{raw_reason}"
            return ("mouse", self._last_control_mode_source, raw_mode)
        if self._sticky_control_mode in ("mouse", "keyboard"):
            self._last_control_mode_source = f"sticky:{self._sticky_control_mode}"
            return (self._sticky_control_mode, self._last_control_mode_source, raw_mode)
        self._last_control_mode_source = f"fallback:{fallback}"
        return (fallback, self._last_control_mode_source, raw_mode)
