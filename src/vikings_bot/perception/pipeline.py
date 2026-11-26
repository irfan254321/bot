"""Unified Pure OpenCV perception pipeline (<8ms CPU): HUD, Coords, Minimap, UI, Gold Rustler & Gold Bag."""

from __future__ import annotations

import time

import cv2

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig, roi_to_pixels
from vikings_bot.perception.coords import CoordReader
from vikings_bot.perception.gold_detector import GoldRustlerDetector
from vikings_bot.perception.gold_loot_detector import GoldBagDetector
from vikings_bot.perception.hud_bars import HudBarReader
from vikings_bot.perception.minimap import MinimapAnalyzer
from vikings_bot.perception.ui_matcher import UiTemplateMatcher
from vikings_bot.state.models import LootDrop, MobTarget, PerceptionSnapshot


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
        self.auto_level_camera_pitch: bool = bool(config.controls.get("auto_level_camera_pitch", False))

    def set_control_mode(self, mode: str) -> None:
        m = str(mode).strip().lower()
        if m in ("mouse", "keyboard"):
            self._control_mode = m

    @property
    def active_backend_desc(self) -> str:
        return "OpenCV-Gold"

    def close(self) -> None:
        return None

    def analyze(
        self,
        frame: FramePacket,
        check_tele_ui: bool = False,
        skip_field_detectors: bool = False,
        cursor_rel_xy: tuple[int, int] | None = None,
        check_interface_mode: bool = True,
        check_repair_ui: bool = False,
    ) -> PerceptionSnapshot:
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

        if not skip_field_detectors:
            t_pf = time.perf_counter()
            ox, oy, px2, py2 = roi_to_pixels(self.playfield_roi, frame.width, frame.height)
            pf_crop = frame.image_bgr[oy:py2, ox:px2]
            shared_pf_hsv = cv2.cvtColor(pf_crop, cv2.COLOR_BGR2HSV) if pf_crop.size > 0 else None
            timings["hsv_conv"] = (time.perf_counter() - t_pf) * 1000.0

            t_m = time.perf_counter()
            mob_list, contested_list, other_players, other_player_boxes = self.mobs.detect_with_context(
                frame, shared_playfield_hsv=shared_pf_hsv
            )
            timings["gold_mobs"] = (time.perf_counter() - t_m) * 1000.0

            t_l = time.perf_counter()
            loot_list, pickup_changed = self.loot.detect(
                frame,
                mobs=mob_list + contested_list,
                other_player_boxes=other_player_boxes,
                shared_playfield_hsv=shared_pf_hsv,
            )
            dead_corpses = self.loot.last_corpse_boxes
            timings["gold_loot"] = (time.perf_counter() - t_l) * 1000.0

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
            camera_pitch_state="ok",
            timings_ms=timings,
        )
