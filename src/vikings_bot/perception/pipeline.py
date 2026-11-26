"""Unified Pure OpenCV perception pipeline (<8ms CPU): HUD, Coords, Minimap, UI, Gold Rustler & Gold Bag."""

from __future__ import annotations

import time

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig, roi_to_pixels
from vikings_bot.perception.coords import CoordReader
from vikings_bot.perception.gold_detector import GoldRustlerDetector
from vikings_bot.perception.gold_loot_detector import GoldBagDetector
from vikings_bot.perception.hud_bars import HudBarReader
from vikings_bot.perception.hud_mask import build_playfield_hud_mask, compute_self_zone
from vikings_bot.perception.minimap import MinimapAnalyzer
from vikings_bot.perception.ui_matcher import UiTemplateMatcher
from vikings_bot.state.models import BotState, LootDrop, MobTarget, PerceptionSnapshot


class PerceptionPipeline:
    """Runs HUD, Coords, Minimap, UI templates, and Pure OpenCV Gold Rustler + Gold Bag extractors."""

    def __init__(self, config: BotConfig) -> None:
        self.hud_bars = HudBarReader(config)
        self.coords = CoordReader(config)
        self.minimap = MinimapAnalyzer(config)
        self.ui_matcher = UiTemplateMatcher(config)
        self.mobs = GoldRustlerDetector(config)
        self.loot = GoldBagDetector(config)
        self.playfield_roi = config.roi("playfield")
        self.ea_roi = config.roi("ea_stat") if "ea_stat" in config.rois else None

        self.control_mode_setting: str = str(config.controls.get("control_mode", "auto")).lower()
        self._control_mode: str = str(config.controls.get("mode", "mouse")).lower()
        self.auto_restore_mouse_mode: bool = bool(config.controls.get("auto_restore_mouse_mode", True))
        self.auto_level_camera_pitch: bool = bool(config.farming.get("auto_camera_pitch", True))
        self.pitch_horizon_min_y: float = float(config.farming.get("pitch_horizon_min_y", 105.0))
        self.pitch_horizon_max_y: float = float(config.farming.get("pitch_horizon_max_y", 300.0))
        self.pitch_sky_ratio_min: float = float(config.farming.get("pitch_sky_ratio_min", 0.12))
        self.pitch_sky_ratio_max: float = float(config.farming.get("pitch_sky_ratio_max", 0.42))
        self.anti_ks_enabled: bool = bool(config.perception.get("anti_ks_enabled", True))

    def _estimate_sky_and_horizon(
        self,
        pf_hsv: np.ndarray,
        ox: int,
        oy: int,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float, str]:
        """Estimate sky_ratio and 768p-normalized horizon_y from upper playfield (<0.08ms CPU)."""
        if pf_hsv.size == 0 or frame_w < 64 or frame_h < 64:
            return (0.25, 190.0, "ok")
        x1 = max(0, int(round(frame_w * 0.16)) - ox)
        x2 = min(pf_hsv.shape[1], int(round(frame_w * 0.88)) - ox)
        y1 = 0
        y2 = min(pf_hsv.shape[0], max(12, int(round(frame_h * 0.67)) - oy))
        if (x2 - x1) < 16 or (y2 - y1) < 12:
            return (0.25, 190.0, "ok")

        sub = pf_hsv[y1:y2:4, x1:x2:4]
        h_ch, s_ch, v_ch = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
        # Bright low-saturation sky/mist OR blue upper sky
        sky = ((v_ch >= 135) & (s_ch <= 75)) | ((h_ch >= 92) & (h_ch <= 132) & (v_ch >= 90))
        sky_ratio = float(sky.mean())
        row_sky = sky.mean(axis=1)
        sky_rows = np.where(row_sky >= 0.35)[0]
        if sky_rows.size > 0:
            full_y = float(oy + y1 + int(sky_rows[-1]) * 4)
            horizon_y = max(0.0, min(768.0, (full_y / float(frame_h)) * 768.0))
        else:
            horizon_y = 0.0

        if horizon_y < self.pitch_horizon_min_y or sky_ratio < self.pitch_sky_ratio_min:
            pstate = "too_down"
        elif horizon_y > self.pitch_horizon_max_y or sky_ratio > self.pitch_sky_ratio_max:
            pstate = "too_up"
        else:
            pstate = "ok"
        return (round(sky_ratio, 4), round(horizon_y, 1), pstate)

    def set_control_mode(self, mode: str) -> None:
        m = str(mode).strip().lower()
        if m in ("mouse", "keyboard"):
            self._control_mode = m

    @property
    def active_backend_desc(self) -> str:
        return "OpenCV-Gold"

    def close(self) -> None:
        return None

    def _probe_combat_target_hp_bar(
        self,
        pf_hsv: np.ndarray,
        ox: int,
        oy: int,
        frame_w: int,
        frame_h: int,
    ) -> tuple[bool, float | None]:
        """Lightweight (<0.2ms) green target HP-bar check during COMBAT_WAIT when GoldRustlerDetector is off."""
        sx = 1.0 if self.mobs.hud_fixed_pixel_ui else (frame_w / 1920.0)
        sy = 1.0 if self.mobs.hud_fixed_pixel_ui else (frame_h / 1080.0)
        cx = frame_w // 2
        cy = frame_h // 2
        rx = int(round(210.0 * sx))
        ry_top = int(round(190.0 * sy))
        ry_bot = int(round(120.0 * sy))
        x1 = max(0, (cx - rx) - ox)
        x2 = min(pf_hsv.shape[1], (cx + rx) - ox)
        y1 = max(0, (cy - ry_top) - oy)
        y2 = min(pf_hsv.shape[0], (cy + ry_bot) - oy)
        if (x2 - x1) < 20 or (y2 - y1) < 8:
            return (False, None)
        sub = pf_hsv[y1:y2, x1:x2]
        g = cv2.inRange(sub, self.mobs.hp_lo, self.mobs.hp_hi)
        if cv2.countNonZero(g) < 12:
            return (False, None)
        n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
        expected_w = max(36.0, 68.0 * sx)
        best_w = 0
        for idx in range(1, n_lbl):
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if 6 <= w <= int(round(95.0 * sx)) and 2 <= h <= max(6, int(round(7.0 * sy))):
                if (w / max(1.0, float(h))) >= 2.0 and (area / max(1.0, float(w * h))) >= 0.50:
                    if w > best_w:
                        best_w = w
        if best_w >= 6:
            return (True, max(0.03, min(1.0, float(best_w) / expected_w)))
        return (False, None)

    def _probe_hit_damage_numbers(
        self,
        pf_hsv: np.ndarray,
        ox: int,
        oy: int,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """Detect local floating orange/yellow damage numbers near character (x=340..685, y=190..465 at 1024x768).

        Uses HSV H=8..35, S>=115, V>=175 and contour height >= 14px so faraway player damage is ignored.
        """
        if pf_hsv.size == 0 or frame_w < 64 or frame_h < 64:
            return False
        sx = 1.0 if self.mobs.hud_fixed_pixel_ui else (frame_w / 1024.0)
        sy = 1.0 if self.mobs.hud_fixed_pixel_ui else (frame_h / 768.0)
        x1 = max(0, int(round(340.0 * sx)) - ox)
        x2 = min(pf_hsv.shape[1], int(round(685.0 * sx)) - ox)
        y1 = max(0, int(round(190.0 * sy)) - oy)
        y2 = min(pf_hsv.shape[0], int(round(465.0 * sy)) - oy)
        if (x2 - x1) < 16 or (y2 - y1) < 16:
            return False

        sub = pf_hsv[y1:y2, x1:x2]
        mask = cv2.inRange(
            sub,
            np.array([8, 115, 175], dtype=np.uint8),
            np.array([35, 255, 255], dtype=np.uint8),
        )
        min_h = max(12, int(round(14.0 * sy)))
        if cv2.countNonZero(mask) < min_h * 2:
            return False
        n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_w = int(round(180.0 * sx))
        max_h = int(round(95.0 * sy))
        for idx in range(1, n_lbl):
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if 3 <= w <= max_w and min_h <= h <= max_h and area >= max(16, min_h):
                return True
        return False

    def _detect_external_damage_points(
        self,
        pf_hsv: np.ndarray,
        hud_mask: np.ndarray,
        ox: int,
        oy: int,
        frame_w: int,
        frame_h: int,
    ) -> tuple[tuple[int, int], ...]:
        """Detect external floating damage numbers across playfield during SCAN_SPOT / SEARCH_MOB (Anti-KS stealth Sin guard).

        Matches HSV H=8..35, S>=95, V>=150 with contour height >= 14px (scaled from 1024x768).
        """
        if not self.anti_ks_enabled or pf_hsv.size == 0 or frame_w < 64 or frame_h < 64:
            return ()
        sx = 1.0 if self.mobs.hud_fixed_pixel_ui else (frame_w / 1024.0)
        sy = 1.0 if self.mobs.hud_fixed_pixel_ui else (frame_h / 768.0)

        mask = cv2.bitwise_and(
            cv2.inRange(
                pf_hsv,
                np.array([8, 95, 150], dtype=np.uint8),
                np.array([35, 255, 255], dtype=np.uint8),
            ),
            hud_mask,
        )
        min_h = max(12, int(round(14.0 * sy)))
        if cv2.countNonZero(mask) < min_h * 2:
            return ()

        n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_w = int(round(180.0 * sx))
        max_h = int(round(95.0 * sy))
        min_area = max(18, int(round(20.0 * sx * sy)))
        cluster_r = 45.0 * sx

        pts: list[tuple[int, int]] = []
        for idx in range(1, n_lbl):
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if not (4 <= w <= max_w and min_h <= h <= max_h and area >= min_area):
                continue
            cx = ox + int(stats[idx, cv2.CC_STAT_LEFT]) + w // 2
            cy = oy + int(stats[idx, cv2.CC_STAT_TOP]) + h // 2
            if any(abs(cx - px) <= cluster_r and abs(cy - py) <= cluster_r for px, py in pts):
                continue
            pts.append((cx, cy))
        return tuple(pts)

    def analyze(
        self,
        frame: FramePacket,
        check_tele_ui: bool = False,
        skip_field_detectors: bool = False,
        cursor_rel_xy: tuple[int, int] | None = None,
        check_interface_mode: bool = True,
        check_repair_ui: bool = False,
        skip_mob_detection: bool = False,
        skip_all_object_detectors: bool = False,
        loot_radius_limit_px: float | None = None,
        current_state: BotState | None = None,
    ) -> PerceptionSnapshot:
        # Hard-gate at perception entry based on FSM state:
        # - COMBAT_WAIT: NEVER run GoldRustlerDetector, GoldBagDetector, or CorpseDetector (THP & Player HP only!)
        # - Non-LOOT states: NEVER run GoldBagDetector so living monster armor never triggers LOOT!
        # - LOOT: NEVER run GoldRustlerDetector; run ONLY GoldBagDetector within <=220px close radius!
        in_combat_wait = current_state == BotState.COMBAT_WAIT or skip_all_object_detectors
        run_loot_detector = (not in_combat_wait) and (
            current_state is None or current_state in (BotState.LOOT, BotState.PAUSED)
        )
        run_mob_detector = (not in_combat_wait) and (not skip_mob_detection) and (
            current_state != BotState.LOOT
        )
        effective_loot_radius = (
            loot_radius_limit_px
            if loot_radius_limit_px is not None
            else (310.0 if current_state in (BotState.LOOT, BotState.PAUSED) else None)
        )
        t_wall0 = time.perf_counter()
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        hp_ratio, mp_ratio = self.hud_bars.read(frame)
        ea_value = self.coords.read_int_roi(frame, self.ea_roi) if self.ea_roi is not None else None
        timings["hud_bars"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        coord_reading = self.coords.read(frame)
        timings["coords"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        minimap_reading = self.minimap.analyze(frame)
        timings["minimap"] = (time.perf_counter() - t0) * 1000.0

        mob_list: tuple[MobTarget, ...] = ()
        contested_list: tuple[MobTarget, ...] = ()
        other_players: tuple[tuple[int, int], ...] = ()
        other_player_boxes: tuple[tuple[int, int, int, int], ...] = ()
        loot_list: tuple[LootDrop, ...] = ()
        dead_corpses: tuple[tuple[int, int, int, int], ...] = ()
        pickup_changed = False
        playfield_luma: float | None = None
        sky_ratio: float | None = None
        horizon_y: float | None = None
        camera_pitch_state: str = "ok"
        combat_target_hp_visible = False
        combat_target_hp_ratio: float | None = None
        hit_damage_visible = False
        external_damage_points: tuple[tuple[int, int], ...] = ()

        if not skip_field_detectors:
            t_pf = time.perf_counter()
            ox, oy, px2, py2 = roi_to_pixels(self.playfield_roi, frame.width, frame.height)
            pf_crop = frame.image_bgr[oy:py2, ox:px2]
            shared_pf_hsv = cv2.cvtColor(pf_crop, cv2.COLOR_BGR2HSV) if pf_crop.size > 0 else None
            if shared_pf_hsv is not None:
                hud_mask = build_playfield_hud_mask(
                    frame.width,
                    frame.height,
                    (ox, oy, px2, py2),
                    hud_fixed_pixel_ui=self.mobs.hud_fixed_pixel_ui,
                )
                v_sub = shared_pf_hsv[::4, ::4, 2]
                m_sub = hud_mask[::4, ::4] > 0
                playfield_luma = float(v_sub[m_sub].mean()) if bool(np.any(m_sub)) else float(v_sub.mean())
                sky_ratio, horizon_y, camera_pitch_state = self._estimate_sky_and_horizon(
                    shared_pf_hsv, ox, oy, frame.width, frame.height
                )
            timings["hsv_conv"] = (time.perf_counter() - t_pf) * 1000.0

            t_m = time.perf_counter()
            if in_combat_wait:
                if shared_pf_hsv is not None:
                    combat_target_hp_visible, combat_target_hp_ratio = self._probe_combat_target_hp_bar(
                        shared_pf_hsv, ox, oy, frame.width, frame.height
                    )
                    hit_damage_visible = self._probe_hit_damage_numbers(
                        shared_pf_hsv, ox, oy, frame.width, frame.height
                    )
                timings["gold_mobs"] = (time.perf_counter() - t_m) * 1000.0
                timings["gold_loot"] = 0.0
            else:
                if not run_mob_detector:
                    if shared_pf_hsv is not None:
                        combat_target_hp_visible, combat_target_hp_ratio = self._probe_combat_target_hp_bar(
                            shared_pf_hsv, ox, oy, frame.width, frame.height
                        )
                        if current_state == BotState.LOOT and self.anti_ks_enabled:
                            (_fx, _fy), s_box = compute_self_zone(
                                frame.width,
                                frame.height,
                                player_center_norm=self.mobs.player_center_norm,
                                hud_fixed_pixel_ui=self.mobs.hud_fixed_pixel_ui,
                            )
                            other_players, other_player_boxes = self.mobs._detect_other_players(
                                shared_pf_hsv,
                                hud_mask,
                                ox,
                                oy,
                                frame.width,
                                frame.height,
                                s_box,
                                [],
                            )
                else:
                    mob_list, contested_list, other_players, other_player_boxes = self.mobs.detect_with_context(
                        frame, shared_playfield_hsv=shared_pf_hsv
                    )
                    if shared_pf_hsv is not None and current_state in (BotState.SEARCH_MOB, BotState.AT_SPOT_IDLE, None):
                        external_damage_points = self._detect_external_damage_points(
                            shared_pf_hsv, hud_mask, ox, oy, frame.width, frame.height
                        )
                timings["gold_mobs"] = (time.perf_counter() - t_m) * 1000.0

                if run_loot_detector:
                    t_l = time.perf_counter()
                    loot_list, pickup_changed = self.loot.detect(
                        frame,
                        mobs=mob_list + contested_list,
                        other_player_boxes=other_player_boxes,
                        shared_playfield_hsv=shared_pf_hsv,
                        max_dist_override_px=effective_loot_radius,
                    )
                    dead_corpses = self.loot.last_corpse_boxes
                    timings["gold_loot"] = (time.perf_counter() - t_l) * 1000.0
                else:
                    timings["gold_loot"] = 0.0

        sign_bbox = None
        menu_bbox = None
        talk_ok_bbox = None
        merchant_sign_bbox = None
        trade_menu_bbox = None
        repair_all_bbox = None
        repair_yes_bbox = None
        inventory_tab_bbox = None
        respawn_dialog_bbox = None

        if check_tele_ui:
            t0 = time.perf_counter()
            sign_bbox = self.ui_matcher.find_addon_keep_sign(frame)
            menu_bbox = self.ui_matcher.find_teleport_menu(frame)
            talk_ok_bbox = self.ui_matcher.find_npc_talk_dialog(frame)
            timings["tele_ui"] = (time.perf_counter() - t0) * 1000.0

        if check_repair_ui:
            t0 = time.perf_counter()
            merchant_sign_bbox = self.ui_matcher.find_merchant_sign(frame)
            trade_menu_bbox = self.ui_matcher.find_trade_menu(frame)
            repair_all_bbox = self.ui_matcher.find_repair_all_btn(frame)
            repair_yes_bbox = self.ui_matcher.find_repair_yes_btn(frame)
            inventory_tab_bbox = self.ui_matcher.find_inventory_tab(frame)
            if talk_ok_bbox is None:
                talk_ok_bbox = self.ui_matcher.find_npc_talk_dialog(frame)
            timings["repair_ui"] = (time.perf_counter() - t0) * 1000.0

        # Opportunistic respawn-popup check ONLY when HP reads 0% AND center screen has a dark modal box (<0.05ms gate!)
        if hp_ratio <= 0.01:
            respawn_dialog_bbox = talk_ok_bbox or repair_yes_bbox
            if respawn_dialog_bbox is None and frame.width >= 64 and frame.height >= 64:
                cy1, cy2 = int(frame.height * 0.35), int(frame.height * 0.68)
                cx1, cx2 = int(frame.width * 0.34), int(frame.width * 0.68)
                center_sub = frame.image_bgr[cy1:cy2:4, cx1:cx2:4]
                if center_sub.size > 0 and float((center_sub.max(axis=2) < 65).mean()) >= 0.08:
                    respawn_dialog_bbox = self.ui_matcher.find_npc_talk_dialog(
                        frame
                    ) or self.ui_matcher.find_repair_yes_btn(frame)

        interface_mode = "unknown"
        detected_ctrl_mode: str | None = None
        if check_interface_mode and (self.auto_restore_mouse_mode or self.control_mode_setting == "auto"):
            t0 = time.perf_counter()
            resolved_mode, _src, interface_mode = self.ui_matcher.detect_control_mode(
                frame,
                cursor_rel_xy=cursor_rel_xy,
                manual_fallback=self._control_mode,
            )
            if interface_mode in ("keyboard_banner", "mouse_banner", "mouse_cursor"):
                detected_ctrl_mode = resolved_mode
            timings["interface_mode"] = (time.perf_counter() - t0) * 1000.0

        timings["total"] = (time.perf_counter() - t_wall0) * 1000.0

        return PerceptionSnapshot(
            timestamp=frame.timestamp,
            frame_width=frame.width,
            frame_height=frame.height,
            hp_ratio=hp_ratio,
            mp_ratio=mp_ratio,
            coords=coord_reading,
            minimap=minimap_reading,
            mobs=mob_list,
            contested_mobs=contested_list,
            other_players=other_players,
            other_player_boxes=other_player_boxes,
            loot=loot_list,
            dead_corpses=dead_corpses,
            pickup_log_changed=pickup_changed,
            ea_value=ea_value,
            addon_keep_sign_bbox=sign_bbox,
            teleport_menu_bbox=menu_bbox,
            npc_talk_dialog_bbox=talk_ok_bbox,
            merchant_sign_bbox=merchant_sign_bbox,
            trade_menu_bbox=trade_menu_bbox,
            repair_all_bbox=repair_all_bbox,
            repair_yes_bbox=repair_yes_bbox,
            inventory_tab_bbox=inventory_tab_bbox,
            respawn_dialog_bbox=respawn_dialog_bbox,
            detector_backend=self.active_backend_desc,
            interface_mode=interface_mode,
            detected_control_mode=detected_ctrl_mode,
            camera_pitch_state=camera_pitch_state,
            sky_ratio=sky_ratio,
            horizon_y=horizon_y,
            playfield_luma=playfield_luma,
            combat_target_hp_visible=combat_target_hp_visible,
            combat_target_hp_ratio=combat_target_hp_ratio,
            hit_damage_visible=hit_damage_visible,
            local_hit_damage_visible=hit_damage_visible,
            external_damage_points=external_damage_points,
            timings_ms=timings,
        )
