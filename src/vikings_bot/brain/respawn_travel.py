"""Auto-Respawn -> Town -> Addon Keep Teleport -> Multi-Waypoint Return-to-Spot controller."""

from __future__ import annotations

import logging
import math

from vikings_bot.brain.navigation import world_bearing_deg
from vikings_bot.config import BotConfig, Waypoint
from vikings_bot.input.actions import (
    ActionIntent,
    ClickScreenIntent,
    DragCameraIntent,
    HoldKeyIntent,
    MoveMouseRelIntent,
    TapKeyIntent,
    TypeChatCommandIntent,
)
from vikings_bot.state.models import BotState, Point2D, WorldState
from vikings_bot.state.tracker import WorldStateTracker, wrap_angle_deg

logger = logging.getLogger(__name__)


class RespawnTravelController:
    """Manages DEAD_RESPAWN -> TELEPORT_TO_SPOT (Addon Keep Teleport) -> TRAVEL_TO_SPOT (waypoints).

    Calibrated directly from Vikings Online 1920x1080 death/town/teleporter screenshots and route videos:
    - Town respawn area: (1075, 2631) r=55
    - Addon Keep Teleport NPC stand point: (1060, 2626) r=6
    - Addon Keep landing gate: (1409, 1779) r=75
    - Supports infinite repeat if killed again before reaching the farming spot.
    """

    def __init__(self, config: BotConfig, tracker: WorldStateTracker) -> None:
        self.config = config
        self.tracker = tracker

        rt = config.respawn_travel
        c = config.controls
        self.enabled: bool = bool(rt.get("enabled", True))
        self.repeat_forever: bool = bool(rt.get("repeat_forever", True))

        cs = rt.get("city_spawn", [1075.0, 2631.0])
        self.city_spawn = Point2D(float(cs[0]), float(cs[1]))
        self.city_radius = float(rt.get("city_radius", 55.0))

        npc = rt.get("addon_teleport_coord", rt.get("npc_stand_coord", [1062.0, 2627.0]))
        self.npc_stand = Point2D(float(npc[0]), float(npc[1]))
        self.npc_arrival_radius = float(rt.get("addon_keep_arrival_radius", rt.get("npc_arrival_radius", 4.0)))

        ak = rt.get("addon_keep_landing", [1409.0, 1773.0])
        self.addon_keep_landing = Point2D(float(ak[0]), float(ak[1]))
        self.addon_keep_radius = float(rt.get("addon_keep_radius", 75.0))

        self.npc_body_offset_y_px = float(rt.get("npc_body_offset_y_px", 38.0))
        self.cursor_toggle_key = str(rt.get("cursor_toggle_key", "tab"))
        self.cursor_fallback_key = str(rt.get("cursor_fallback_key", "esc"))
        self.mode_toggle_key = str(rt.get("mode_toggle_key", "esc"))
        self.respawn_auto_wait_sec = float(rt.get("respawn_auto_wait_sec", 1.5))
        self.respawn_fallback_key = str(rt.get("respawn_fallback_key", "space"))
        self.teleport_load_timeout_sec = float(rt.get("teleport_load_timeout_sec", 12.0))
        self.travel_arrival_radius = float(rt.get("travel_arrival_radius", 13.0))
        self.travel_final_radius = float(rt.get("travel_final_radius", 12.0))

        self.control_mode = str(c.get("mode", "keyboard")).lower()
        # All town/repair/teleport/travel walking is strictly 100% Keyboard Mode (WASD); mouse is ONLY toggled via Tab for NPC/UI clicks.
        self.mouse_only_travel: bool = False
        self.camera_btn: str = str(c.get("camera_rotate_button", "middle"))
        self.player_center_norm = config.player_center_norm()
        self.playfield_roi = config.roi("playfield")
        self.forward_key = str(c.get("move_forward_key", "w"))
        self.back_key = str(c.get("move_back_key", "s"))
        self.strafe_left_key = str(c.get("strafe_left_key", "a"))
        self.strafe_right_key = str(c.get("strafe_right_key", "d"))
        self.jump_key = str(c.get("attack_combo_key", "space"))
        self.yaw_px_per_deg = float(c.get("kb_yaw_px_per_deg", 2.5))
        self.max_turn_px = int(c.get("kb_max_turn_px", 110))

        # Internal dead-respawn double-click state
        self._dead_clicks_in_burst: int = 0

        # Internal teleport sub-state
        self._tele_phase: str = "WALK_TO_NPC"  # WALK_TO_NPC | FIND_SIGN | WAIT_MENU | WAIT_LOAD
        self._tele_phase_ts: float = 0.0
        self._tele_click_attempts: int = 0
        self._last_tele_npc_click_xy: tuple[int, int] | None = None
        self._talk_dismiss_attempts: int = 0
        self._cursor_toggled_by_us: bool = False
        self._cursor_ready_at: float = 0.0
        self._sign_missing_since: float | None = None
        self._mode_switched_to_kb: bool = False
        self._active_cursor_key: str = self.cursor_toggle_key

        # Internal travel unstuck state
        self._travel_unstuck_until: float = 0.0
        self._travel_unstuck_count: int = 0
        self._travel_unstuck_dir: str = "a"

        # Periodic Auto-Repair config & state (/kill -> Beginner Armor Merchant (1094,2643) -> Trade -> Repair All -> Yes)
        rep = config.repair
        self.repair_enabled: bool = bool(rep.get("enabled", True))
        self.repair_interval_sec: float = max(30.0, float(rep.get("interval_minutes", 80.0)) * 60.0)
        self.suicide_command: str = str(rep.get("suicide_command", "/kill"))
        self.suicide_wait_sec: float = float(rep.get("suicide_wait_sec", 18.0))
        self.suicide_retry_sec: float = float(rep.get("suicide_retry_sec", 22.0))
        self.npc_click_hold_sec: float = max(0.04, float(rep.get("npc_click_hold_sec", 0.10)))
        self.close_dialog_with_esc: bool = bool(rep.get("close_dialog_with_esc", False))
        self.use_tab_cursor: bool = bool(rep.get("use_tab_cursor", True))
        ms_c = rep.get("merchant_stand_coord", [1094.0, 2644.0])
        self.merchant_stand = Point2D(float(ms_c[0]), float(ms_c[1]))
        self.merchant_arrival_radius: float = float(rep.get("merchant_arrival_radius", 4.5))
        hub_c = rep.get("town_hub_waypoint", [1075.0, 2631.0])
        self.town_hub = Point2D(float(hub_c[0]), float(hub_c[1]))
        self.hud_fixed_pixel_ui: bool = bool(config.perception.get("hud_fixed_pixel_ui", False))

        self._recall_cmd_sent_at: float | None = None
        self._recall_attempts: int = 0
        self._repair_phase: str = "WALK_TO_MERCHANT"
        self._repair_phase_ts: float = 0.0
        self._repair_click_attempts: int = 0
        self._last_merchant_npc_click_xy: tuple[int, int] | None = None
        self._repair_confirm_retries: int = 0
        self._repair_talk_dismiss_attempts: int = 0
        self._repair_cursor_toggled: bool = False
        self._repair_cursor_ready_at: float = 0.0
        self._repair_sign_missing_since: float | None = None
        self._repair_mode_switched_to_kb: bool = False
        self._close_ui_attempts: int = 0
        self._close_inv_attempts: int = 0

    def release_all_move_keys(self, reason: str = "respawn_stop_move") -> list[ActionIntent]:
        return [
            HoldKeyIntent(key=self.forward_key, hold=False, reason=reason),
            HoldKeyIntent(key=self.back_key, hold=False, reason=reason),
            HoldKeyIntent(key=self.strafe_left_key, hold=False, reason=reason),
            HoldKeyIntent(key=self.strafe_right_key, hold=False, reason=reason),
        ]

    def reset_teleport_state(self, now: float) -> None:
        self._dead_clicks_in_burst = 0
        self._tele_phase = "WALK_TO_NPC"
        self._tele_phase_ts = now
        self._tele_click_attempts = 0
        self._last_tele_npc_click_xy = None
        self._talk_dismiss_attempts = 0
        self._cursor_toggled_by_us = False
        self._cursor_ready_at = 0.0
        self._sign_missing_since = None
        self._mode_switched_to_kb = False
        self._active_cursor_key = self.cursor_toggle_key

    def reset_travel_state(self) -> None:
        self._travel_unstuck_until = 0.0
        self._travel_unstuck_count = 0

    def reset_repair_recall_state(self, now: float) -> None:
        self._dead_clicks_in_burst = 0
        self._recall_cmd_sent_at = None
        self._recall_attempts = 0
        self.reset_repair_npc_state(now)

    def reset_repair_npc_state(self, now: float) -> None:
        self._repair_phase = "WALK_TO_MERCHANT"
        self._repair_phase_ts = now
        self._repair_click_attempts = 0
        self._last_merchant_npc_click_xy = None
        self._repair_confirm_retries = 0
        self._repair_talk_dismiss_attempts = 0
        self._repair_cursor_toggled = False
        self._repair_cursor_ready_at = 0.0
        self._repair_sign_missing_since = None
        self._repair_mode_switched_to_kb = False
        self._close_ui_attempts = 0
        self._close_inv_attempts = 0

    def is_in_city(self, coords: Point2D | None) -> bool:
        if coords is None:
            return False
        if 1020.0 <= coords.x <= 1160.0 and 2560.0 <= coords.y <= 2700.0:
            return True
        return coords.distance_to(self.city_spawn) <= self.city_radius

    def is_near_addon_keep(self, coords: Point2D | None) -> bool:
        if coords is None:
            return False
        return coords.distance_to(self.addon_keep_landing) <= self.addon_keep_radius

    def choose_starting_travel_index(self, coords: Point2D | None, travel_waypoints: list[Waypoint]) -> int:
        """Pick the nearest forward travel waypoint when resuming mid-route outside town."""
        if coords is None or not travel_waypoints:
            return 0
        best_idx = 0
        best_dist = float("inf")
        for idx, wp in enumerate(travel_waypoints):
            d = coords.distance_to(Point2D(wp.x, wp.y))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        # If we're already right on that waypoint and there's a next one, aim for the next one
        if best_dist <= self.travel_arrival_radius and best_idx + 1 < len(travel_waypoints):
            return best_idx + 1
        return best_idx

    def _is_valid_popup_near_npc(
        self,
        menu_bbox: tuple[int, int, int, int],
        anchor_xy: tuple[int, int] | None,
        fw: int,
        fh: int,
    ) -> bool:
        """Reject sky/top false positives (y < 140 on 1024x768, scaled) and popups far from last NPC click."""
        mx, my, mw, mh = menu_bbox
        sx = 1.0 if self.hud_fixed_pixel_ui else (fw / 1024.0)
        sy = 1.0 if self.hud_fixed_pixel_ui else (fh / 768.0)
        if my < int(round(140.0 * sy)):
            return False
        if anchor_xy is not None:
            ax, ay = anchor_xy
            mcx = mx + mw // 2
            mcy = my + mh // 2
            if abs(mcx - ax) > int(round(180.0 * sx)):
                return False
            if (mcy - ay) < int(round(-240.0 * sy)) or (mcy - ay) > int(round(200.0 * sy)):
                return False
        return True

    def _steer_keyboard_toward(
        self,
        world: WorldState,
        target: Point2D,
        target_name: str,
        now: float,
    ) -> list[ActionIntent]:
        """Fast, smooth closed-loop Keyboard Mode steering toward a world coordinate."""
        coords = world.coords_smoothed
        if coords is None:
            return []
        dist = coords.distance_to(target)
        bearing = world_bearing_deg(coords, target)
        rel_angle = wrap_angle_deg(bearing - world.camera_heading_deg)

        # If compass heading isn't calibrated from real motion delta yet, walk straight ~2 units first
        if world.heading_source != "delta":
            return [
                HoldKeyIntent(
                    key=self.forward_key,
                    hold=True,
                    reason=f"travel_calibrate->{target_name} d={dist:.1f}",
                    move_rel_angle_deg=0.0,
                )
            ]

        intents: list[ActionIntent] = []
        turn_thresh = 8.0 if dist < 10.0 else 10.0
        if abs(rel_angle) > turn_thresh and self.tracker.is_ready("travel_yaw", now):
            raw_dx = int(round(-rel_angle * self.yaw_px_per_deg * 0.58))
            dx = max(-self.max_turn_px, min(self.max_turn_px, raw_dx))
            if abs(dx) >= 4:
                est_yaw = -dx / max(1.0, self.yaw_px_per_deg)
                self.tracker.set_cooldown("travel_yaw", now, 0.08)
                intents.append(
                    MoveMouseRelIntent(
                        dx=dx,
                        reason=f"travel_yaw->{target_name} err={rel_angle:.0f}",
                        est_yaw_deg=est_yaw,
                        steps=3,
                        step_delay=0.002,
                    )
                )

        # Tighter heading cone when dist < 8 units + short-step pulsing so character NEVER overshoots (1062, 2627)!
        walk_cone = 24.0 if dist < 8.0 else 50.0
        if abs(rel_angle) <= walk_cone:
            if dist < 8.0:
                intents.append(HoldKeyIntent(key=self.forward_key, hold=False, reason="travel_short_step_brake"))
                if self.tracker.is_ready("travel_short_step", now):
                    self.tracker.set_cooldown("travel_short_step", now, 0.26)
                    intents.append(
                        TapKeyIntent(
                            key=self.forward_key,
                            hold_sec=0.14,
                            reason=f"travel_short_step->{target_name} d={dist:.1f}",
                        )
                    )
            else:
                intents.append(
                    HoldKeyIntent(
                        key=self.forward_key,
                        hold=True,
                        reason=f"travel->{target_name} d={dist:.1f}",
                        move_rel_angle_deg=0.0,
                    )
                )
        else:
            intents.append(HoldKeyIntent(key=self.forward_key, hold=False, reason="travel_turn_first"))
        return intents

    def step_dead_respawn(
        self,
        world: WorldState,
        now: float,
        elapsed: float,
    ) -> tuple[list[ActionIntent], BotState | None, str]:
        """Wait for automatic town respawn (or tap fallback key after respawn_auto_wait_sec)."""
        intents = self.release_all_move_keys("dead_wait_respawn")

        # Keep jump acceptance armed while dead so the sudden teleport from spot -> town is immediately accepted
        self.tracker.accept_teleport_jump()

        alive = world.hp_ratio >= 0.25 and not world.is_dead
        if alive and self.is_in_city(world.coords_smoothed):
            self.reset_teleport_state(now)
            return (intents, BotState.TELEPORT_TO_SPOT, "respawned_in_town")

        # Edge case: revived directly near Addon Keep (e.g. saved bind stone / resurrection)
        if alive and self.is_near_addon_keep(world.coords_smoothed):
            self.reset_travel_state()
            self.tracker.travel_waypoint_index = 0
            return (intents, BotState.TRAVEL_TO_SPOT, "revived_at_addon_keep")

        # Edge case: resurrected directly at the farming spot by a party healer!
        if alive and elapsed >= 1.5 and world.coords_smoothed is not None:
            return (intents, BotState.SEARCH_MOB, "revived_in_field")

        # Vikings Online dead screen (e.g. PK'd / killed): LEFT-CLICK 2x to revive/respawn back to town!
        # Send click #1, then 0.22s later click #2, repeating the 2x burst every 2.2s until alive/in town.
        if elapsed >= self.respawn_auto_wait_sec and self.tracker.is_ready("respawn_click", now):
            cx = world.snapshot.frame_width // 2
            cy = world.snapshot.frame_height // 2
            self._dead_clicks_in_burst += 1
            if self._dead_clicks_in_burst == 1:
                self.tracker.set_cooldown("respawn_click", now, 0.22)
                intents.append(
                    ClickScreenIntent(x=cx, y=cy, button="left", reason="dead_respawn_lclick_1")
                )
            else:
                self._dead_clicks_in_burst = 0
                self.tracker.set_cooldown("respawn_click", now, 2.2)
                intents.append(
                    ClickScreenIntent(x=cx, y=cy, button="left", reason="dead_respawn_lclick_2")
                )

        # Secondary fallback key after 7.5s if clicks alone didn't dismiss a dialog
        if (
            elapsed >= max(7.5, self.respawn_auto_wait_sec + 6.0)
            and self.respawn_fallback_key
            and self.tracker.is_ready("respawn_key", now)
        ):
            self.tracker.set_cooldown("respawn_key", now, 4.0)
            intents.append(
                TapKeyIntent(key=self.respawn_fallback_key, hold_sec=0.05, reason="respawn_fallback_tap")
            )

        return (intents, None, "")

    def step_repair_recall(
        self,
        world: WorldState,
        now: float,
        elapsed: float,
    ) -> tuple[list[ActionIntent], BotState | None, str]:
        """Freeze character completely, wait 0.25s for AHK E-spam interlock, send '/kill', and stand still until in Green Wales."""
        intents = self.release_all_move_keys("repair_recall_wait")
        self.tracker.accept_teleport_jump()

        alive = world.hp_ratio >= 0.25 and not world.is_dead
        if alive and self.is_in_city(world.coords_smoothed):
            self.reset_repair_npc_state(now)
            return (intents, BotState.REPAIR_AT_NPC, "arrived_in_town_for_repair")

        # Hard freeze: wait >= 0.25s after entering REPAIR_RECALL so AHK E-spam & prior actions settle cleanly.
        # NEVER send ClickScreenIntent, MoveMouseRelIntent, WASD, or Esc during /kill countdown!
        if elapsed < 0.25:
            return (intents, None, "")

        need_send = self._recall_cmd_sent_at is None or (
            (now - self._recall_cmd_sent_at) >= self.suicide_retry_sec and self._recall_attempts < 3
        )
        if need_send and not world.is_dead:
            self._recall_cmd_sent_at = now
            self._recall_attempts += 1
            intents.append(
                TypeChatCommandIntent(
                    command=self.suicide_command,
                    reason=f"repair_suicide_cmd#{self._recall_attempts}",
                )
            )
            logger.info(
                "Auto-Repair Recall #%d: sending '%s' -> standing still ~%.0fs for town teleport...",
                self._recall_attempts,
                self.suicide_command,
                self.suicide_wait_sec,
            )

        return (intents, None, "")

    def step_repair_at_npc(
        self,
        world: WorldState,
        now: float,
    ) -> tuple[list[ActionIntent], BotState | None, str]:
        """Walk to Beginner Armor Merchant (1094,2643) -> Trade -> Repair All -> Yes -> Close UI -> Teleport."""
        coords = world.coords_smoothed
        fw = world.snapshot.frame_width
        fh = world.snapshot.frame_height
        scale_y = fh / 1080.0
        ui_sx = 1.0 if self.hud_fixed_pixel_ui else (fw / 1920.0)
        ui_sy = 1.0 if self.hud_fixed_pixel_ui else (fh / 1080.0)

        talk_ok = world.snapshot.npc_talk_dialog_bbox
        # Guard against accidentally clicking 'Talk' instead of 'Trade' ("All my armors are from ancient Vikings... [OK]")
        if talk_ok is not None and self._repair_phase in ("FIND_MERCHANT_SIGN", "WAIT_TRADE_MENU"):
            intents = self.release_all_move_keys("dismiss_merchant_talk_dialog")
            self._repair_phase = "FIND_MERCHANT_SIGN"
            self._repair_phase_ts = now
            self._repair_sign_missing_since = None
            if self.tracker.is_ready("repair_talk_dismiss", now):
                self.tracker.set_cooldown("repair_talk_dismiss", now, 0.45)
                self._repair_talk_dismiss_attempts += 1
                if self._repair_talk_dismiss_attempts % 2 == 1:
                    ox, oy, ow, oh = talk_ok
                    ok_cx = max(20, min(fw - 20, ox + ow // 2))
                    ok_cy = max(20, min(fh - 20, oy + oh // 2))
                    intents.append(
                        ClickScreenIntent(
                            x=ok_cx,
                            y=ok_cy,
                            button="left",
                            reason=f"click_merchant_talk_ok#{self._repair_talk_dismiss_attempts}",
                            no_jitter=True,
                        )
                    )
                    logger.info(
                        "Accidental Merchant 'Talk' popup detected -> clicking [OK] at (%d, %d) & retrying Trade",
                        ok_cx,
                        ok_cy,
                    )
                else:
                    intents.append(
                        TapKeyIntent(
                            key="enter",
                            hold_sec=0.05,
                            reason=f"dismiss_merchant_talk_enter#{self._repair_talk_dismiss_attempts}",
                        )
                    )
                    logger.info("Merchant 'Talk' popup still visible -> tapping ENTER & retrying Trade")
            return (intents, None, "")
        if self._repair_phase in ("FIND_MERCHANT_SIGN", "WAIT_TRADE_MENU"):
            self._repair_talk_dismiss_attempts = 0

        # Phase 1: Walk from town spawn (1075,2631) -> Beginner Armor Merchant (1094,2644) strictly in Keyboard Mode (WASD)
        if self._repair_phase == "WALK_TO_MERCHANT":
            intents = []
            if coords is None:
                return (intents, None, "")
            dist_m = coords.distance_to(self.merchant_stand)
            if dist_m <= self.merchant_arrival_radius:
                intents.extend(self.release_all_move_keys("arrived_at_merchant"))
                self._repair_phase = "FIND_MERCHANT_SIGN"
                self._repair_phase_ts = now
                logger.info("Arrived at Beginner Armor Merchant (%.0f, %.0f)", coords.x, coords.y)
                return (intents, None, "")

            if world.is_stuck and self.tracker.is_ready("repair_unstick", now):
                self.tracker.set_cooldown("repair_unstick", now, 0.8)
                self.tracker.clear_stuck(reset_consecutive=True)
                intents.extend(self.release_all_move_keys("repair_walk_unstick"))
                intents.append(TapKeyIntent(key=self.strafe_left_key, hold_sec=0.35, reason="repair_sidestep"))
                intents.append(TapKeyIntent(key=self.jump_key, hold_sec=0.05, reason="repair_jump"))
                return (intents, None, "")

            intents.extend(self._steer_keyboard_toward(world, self.merchant_stand, "armor_merchant", now))
            return (intents, None, "")

        # Phase 2: Find 'Beginner Armor Merchant' sign, toggle Tab cursor, and left-click NPC
        if self._repair_phase == "FIND_MERCHANT_SIGN":
            intents = self.release_all_move_keys("find_merchant_sign")
            if world.snapshot.repair_all_bbox is not None:
                self._repair_phase = "CLICK_REPAIR_ALL"
                self._repair_phase_ts = now
                return (intents, None, "")
            # NEVER shortcut into trade_menu before NPC has actually been clicked (_repair_click_attempts > 0)!
            if (
                self._repair_click_attempts > 0
                and world.snapshot.trade_menu_bbox is not None
                and self._is_valid_popup_near_npc(
                    world.snapshot.trade_menu_bbox, self._last_merchant_npc_click_xy, fw, fh
                )
            ):
                self._repair_phase = "WAIT_TRADE_MENU"
                self._repair_phase_ts = now
                return (intents, None, "")

            sign = world.snapshot.merchant_sign_bbox
            if sign is not None:
                self._repair_sign_missing_since = None
                sx, sy, sw, sh = sign
                sign_cx = sx + sw // 2
                sign_cy = sy + sh // 2
                if (fw * 0.12) <= sign_cx <= (fw * 0.88):
                    if not self.tracker.is_ready("repair_npc_click", now):
                        return (intents, None, "")
                    next_attempt = self._repair_click_attempts + 1
                    toggle_key = self.cursor_toggle_key
                    if not self._repair_cursor_toggled:
                        self._active_cursor_key = toggle_key
                        self._repair_cursor_toggled = True
                        self._repair_cursor_ready_at = now + 0.18
                        intents.append(
                            TapKeyIntent(key=toggle_key, hold_sec=0.05, reason=f"repair_show_cursor_{toggle_key}")
                        )
                        return (intents, None, "")
                    if now < self._repair_cursor_ready_at:
                        return (intents, None, "")

                    self._repair_click_attempts = next_attempt
                    click_x = max(20, min(fw - 20, sign_cx))
                    click_y = max(20, min(fh - 40, int(round(sign_cy + self.npc_body_offset_y_px * scale_y))))
                    self._last_merchant_npc_click_xy = (click_x, click_y)
                    self.tracker.set_cooldown("repair_npc_click", now, 1.6)
                    intents.append(
                        ClickScreenIntent(
                            x=click_x,
                            y=click_y,
                            button="left",
                            hold_sec=self.npc_click_hold_sec,
                            reason=f"click_merchant_npc#{self._repair_click_attempts}",
                            no_jitter=True,
                        )
                    )
                    self._repair_phase = "WAIT_TRADE_MENU"
                    self._repair_phase_ts = now
                    return (intents, None, "")

            if self._repair_sign_missing_since is None:
                self._repair_sign_missing_since = now
            if self._repair_cursor_toggled:
                if (now - self._repair_sign_missing_since) < 1.2:
                    return (intents, None, "")
                intents.append(
                    TapKeyIntent(key=self._active_cursor_key, hold_sec=0.04, reason="repair_hide_cursor_to_scan")
                )
                self._repair_cursor_toggled = False
            if self.tracker.is_ready("repair_scan_turn", now):
                self.tracker.set_cooldown("repair_scan_turn", now, 0.24)
                dx = 175
                intents.append(
                    MoveMouseRelIntent(
                        dx=dx,
                        reason="scan_for_merchant_sign",
                        est_yaw_deg=-dx / max(1.0, self.yaw_px_per_deg),
                        steps=5,
                        step_delay=0.002,
                    )
                )
            return (intents, None, "")

        # Phase 3: Click 'Trade' option on the 'Talk / Trade' popup
        if self._repair_phase == "WAIT_TRADE_MENU":
            intents = self.release_all_move_keys("wait_trade_menu")
            if world.snapshot.repair_all_bbox is not None:
                self._repair_phase = "CLICK_REPAIR_ALL"
                self._repair_phase_ts = now
                return (intents, None, "")
            tmenu = world.snapshot.trade_menu_bbox
            if tmenu is not None and self._is_valid_popup_near_npc(tmenu, self._last_merchant_npc_click_xy, fw, fh):
                mx, my, mw, mh = tmenu
                trade_x = mx + mw // 2
                trade_y = my + int(round(mh * 0.74))
                self._repair_phase = "CLICK_REPAIR_ALL"
                self._repair_phase_ts = now
                intents.append(
                    ClickScreenIntent(
                        x=trade_x,
                        y=trade_y,
                        button="left",
                        hold_sec=self.npc_click_hold_sec,
                        reason="click_trade_option",
                        no_jitter=True,
                    )
                )
                logger.info("Clicked 'Trade' option at (%d, %d)", trade_x, trade_y)
                return (intents, None, "")

            if (now - self._repair_phase_ts) >= 2.2:
                if coords is not None and coords.distance_to(self.merchant_stand) > (self.merchant_arrival_radius + 4.0):
                    self._repair_phase = "WALK_TO_MERCHANT"
                else:
                    self._repair_phase = "FIND_MERCHANT_SIGN"
                self._repair_phase_ts = now
            return (intents, None, "")

        # Phase 4: Click 'Repair All' inside the open Merchant Shop window
        if self._repair_phase == "CLICK_REPAIR_ALL":
            intents = self.release_all_move_keys("click_repair_all")
            if world.snapshot.repair_yes_bbox is not None:
                self._repair_phase = "CONFIRM_REPAIR"
                self._repair_phase_ts = now
                return (intents, None, "")
            rbtn = world.snapshot.repair_all_bbox
            if rbtn is not None and self.tracker.is_ready("repair_all_click", now):
                rx, ry, rw, rh = rbtn
                rcx = rx + rw // 2
                rcy = ry + rh // 2
                self.tracker.set_cooldown("repair_all_click", now, 0.8)
                self._repair_phase = "CONFIRM_REPAIR"
                self._repair_phase_ts = now
                intents.append(
                    ClickScreenIntent(
                        x=rcx,
                        y=rcy,
                        button="left",
                        hold_sec=self.npc_click_hold_sec,
                        reason="click_repair_all_btn",
                        no_jitter=True,
                    )
                )
                logger.info("Clicked 'Repair All' button at (%d, %d)", rcx, rcy)
                return (intents, None, "")

            if (now - self._repair_phase_ts) >= 2.8:
                self._repair_phase = "FIND_MERCHANT_SIGN"
                self._repair_phase_ts = now
            return (intents, None, "")

        # Phase 5: Confirm '[ Yes ]' on the repair gold cost modal (or '[ OK ]' if already 100% durability)
        if self._repair_phase == "CONFIRM_REPAIR":
            intents = self.release_all_move_keys("confirm_repair_yes")
            ybtn = world.snapshot.repair_yes_bbox
            if ybtn is not None:
                yx, yy, yw, yh = ybtn
                ycx = yx + yw // 2
                ycy = yy + yh // 2
                self._repair_phase = "CLOSE_SHOP_INV"
                self._repair_phase_ts = now
                self._close_ui_attempts = 0
                self.tracker.set_cooldown("repair_close_ui", now, 0.35)
                intents.append(
                    ClickScreenIntent(
                        x=ycx,
                        y=ycy,
                        button="left",
                        hold_sec=self.npc_click_hold_sec,
                        reason="click_repair_yes_btn",
                        no_jitter=True,
                    )
                )
                logger.info("Confirmed Repair '[Yes]' at (%d, %d)!", ycx, ycy)
                return (intents, None, "")

            # Branch 2: 'No equipment to repair.' message/dialog -> equipment already 100% repaired!
            if world.snapshot.no_equipment_repair_bbox is not None:
                self._repair_phase = "CLOSE_SHOP_INV"
                self._repair_phase_ts = now
                self._close_ui_attempts = 0
                self.tracker.set_cooldown("repair_close_ui", now, 0.35)
                logger.info("'No equipment to repair.' detected -> treating repair as complete & closing shop")
                return (intents, None, "")

            if talk_ok is not None:
                ox, oy, ow, oh = talk_ok
                self._repair_phase = "CLOSE_SHOP_INV"
                self._repair_phase_ts = now
                self._close_ui_attempts = 0
                self.tracker.set_cooldown("repair_close_ui", now, 0.35)
                intents.append(
                    ClickScreenIntent(
                        x=ox + ow // 2,
                        y=oy + oh // 2,
                        button="left",
                        hold_sec=self.npc_click_hold_sec,
                        reason="click_no_repair_needed_ok",
                        no_jitter=True,
                    )
                )
                return (intents, None, "")

            if (now - self._repair_phase_ts) >= 1.5:
                if self._repair_confirm_retries < 1 and world.snapshot.repair_all_bbox is not None:
                    self._repair_confirm_retries += 1
                    self._repair_phase = "CLICK_REPAIR_ALL"
                    self._repair_phase_ts = now
                else:
                    self._repair_phase = "CLOSE_SHOP_INV"
                    self._repair_phase_ts = now
                    self._close_ui_attempts = 0
            return (intents, None, "")

        # Phase 6: Close Shop & Inventory at once via 1x ESC (no [x] clicks, no 'i', max 1 retry if still open!)
        if self._repair_phase == "CLOSE_SHOP_INV":
            intents = self.release_all_move_keys("close_shop_inv")
            if not self.tracker.is_ready("repair_close_ui", now):
                return (intents, None, "")

            rbtn = world.snapshot.repair_all_bbox
            itab = world.snapshot.inventory_tab_bbox
            ui_still_open = (rbtn is not None) or (itab is not None)

            if self._close_ui_attempts == 0:
                self._close_ui_attempts = 1
                self.tracker.set_cooldown("repair_close_ui", now, 0.40)
                intents.append(
                    TapKeyIntent(
                        key="esc",
                        hold_sec=0.05,
                        reason="close_shop_inv_esc",
                    )
                )
                logger.info("Closing Shop & Inventory windows via 1x ESC")
                return (intents, None, "")

            if self._close_ui_attempts == 1 and ui_still_open:
                self._close_ui_attempts = 2
                self.tracker.set_cooldown("repair_close_ui", now, 0.40)
                intents.append(
                    TapKeyIntent(
                        key="esc",
                        hold_sec=0.05,
                        reason="close_shop_inv_esc_retry",
                    )
                )
                logger.info("Shop/Inventory still visible -> sending max 1x retry ESC")
                return (intents, None, "")

            # Windows closed! Do NOT send any extra ESC or ClickScreenIntent; walk WASD straight to hub -> Addon Keep Teleport.
            self._repair_cursor_toggled = False
            self._repair_phase = "WALK_TO_HUB"
            self._repair_phase_ts = now
            return (intents, None, "")

        # Phase 7: Walk back via central plaza (1075,2631) in 100% Keyboard Mode (WASD)
        if self._repair_phase == "WALK_TO_HUB":
            intents = []
            if coords is None or coords.distance_to(self.town_hub) <= 7.0 or (now - self._repair_phase_ts) >= 8.0:
                intents.extend(self.release_all_move_keys("repair_hub_reached"))
                self.reset_teleport_state(now)
                logger.info("Equipment repair complete -> proceeding to Addon Keep Teleport (1066, 2627)!")
                return (intents, BotState.TELEPORT_TO_SPOT, "equipment_repaired_heading_to_teleport")

            intents.extend(self._steer_keyboard_toward(world, self.town_hub, "town_hub", now))
            return (intents, None, "")

        return ([], None, "")

    def step_teleport_to_spot(
        self,
        world: WorldState,
        now: float,
    ) -> tuple[list[ActionIntent], BotState | None, str]:
        """Walk from town spawn to Addon Keep Teleport NPC (1066,2627) in Keyboard Mode, Tab-click NPC -> Teleport, wait for Addon Keep."""
        coords = world.coords_smoothed
        fw = world.snapshot.frame_width
        fh = world.snapshot.frame_height
        scale_y = fh / 1080.0

        # Already landed at Ardun Keep (1409, 1773)!
        # Ensure we remain strictly in Keyboard Mode (WASD) with free cursor closed:
        # - If interface_mouse_banner is detected (accidental Mouse Mode) -> tap Esc once to return to Keyboard Mode.
        # - Else if free cursor is still active (mouse_cursor) -> tap Tab once to close it.
        # - NEVER send Esc blindly!
        if self.is_near_addon_keep(coords):
            intents = self.release_all_move_keys("landed_addon_keep")
            was_cursor_open = self._cursor_toggled_by_us
            self._cursor_toggled_by_us = False
            self._sign_missing_since = None
            self._talk_dismiss_attempts = 0
            self.reset_travel_state()
            self.tracker.travel_waypoint_index = 0
            if was_cursor_open and world.snapshot.interface_mode == "mouse_cursor":
                intents.append(
                    TapKeyIntent(
                        key=self.cursor_toggle_key,
                        hold_sec=0.04,
                        reason="close_cursor_after_tele_tab",
                    )
                )
            return (intents, BotState.TRAVEL_TO_SPOT, "teleported_to_addon_keep")

        # Accidental 'Talk' NPC modal dialog guard ("Hi there! ... [ OK ]"):
        # Works across ALL teleport phases (including WAIT_LOAD so we NEVER sit frozen for 12s!).
        # Attempt #1 (odd) clicks the exact [ OK ] button directly (zero chance of opening chat!);
        # Attempt #2 (even, fallback if cursor wasn't visible) taps ENTER to dismiss immediately.
        talk_ok = world.snapshot.npc_talk_dialog_bbox
        if talk_ok is not None:
            intents = self.release_all_move_keys("dismiss_npc_talk_dialog")
            self._tele_phase = "FIND_SIGN"
            self._tele_phase_ts = now
            self._sign_missing_since = None
            if self.tracker.is_ready("tele_talk_dismiss", now):
                self.tracker.set_cooldown("tele_talk_dismiss", now, 0.45)
                self._talk_dismiss_attempts += 1
                if self._talk_dismiss_attempts % 2 == 1:
                    ox, oy, ow, oh = talk_ok
                    ok_cx = max(20, min(fw - 20, ox + ow // 2))
                    ok_cy = max(20, min(fh - 20, oy + oh // 2))
                    intents.append(
                        ClickScreenIntent(
                            x=ok_cx,
                            y=ok_cy,
                            button="left",
                            reason=f"click_npc_talk_ok#{self._talk_dismiss_attempts}",
                            no_jitter=True,
                        )
                    )
                    logger.info(
                        "Detected accidental NPC Talk dialog -> clicking [OK] at (%d, %d) & retrying Teleport",
                        ok_cx,
                        ok_cy,
                    )
                else:
                    intents.append(
                        TapKeyIntent(
                            key="enter",
                            hold_sec=0.05,
                            reason=f"dismiss_npc_talk_enter#{self._talk_dismiss_attempts}",
                        )
                    )
                    logger.info("NPC Talk dialog still visible -> tapping ENTER fallback & retrying Teleport")
            return (intents, None, "")
        self._talk_dismiss_attempts = 0

        # Phase 1: Walk close to Addon Keep Teleport NPC at (1062, 2627) strictly in Keyboard Mode (WASD)
        # NEVER stop early just because the sign is visible from across the street; MUST reach <= 4.0 units!
        if self._tele_phase == "WALK_TO_NPC":
            intents: list[ActionIntent] = []
            if coords is None:
                return (intents, None, "")

            dist_npc = coords.distance_to(self.npc_stand)
            if dist_npc <= self.npc_arrival_radius:
                intents.extend(self.release_all_move_keys("arrived_at_tele_npc"))
                self._tele_phase = "FIND_SIGN"
                self._tele_phase_ts = now
                logger.info("Arrived right in front of Addon Keep Teleport NPC at (%.0f, %.0f) dist=%.1f", coords.x, coords.y, dist_npc)
                return (intents, None, "")

            # Handle obstacle stuck while walking to NPC (e.g. wooden fence edge)
            if world.is_stuck and self.tracker.is_ready("tele_unstick", now):
                self.tracker.set_cooldown("tele_unstick", now, 0.8)
                self.tracker.clear_stuck(reset_consecutive=True)
                intents.extend(self.release_all_move_keys("tele_walk_unstick"))
                intents.append(TapKeyIntent(key=self.strafe_left_key, hold_sec=0.35, reason="tele_sidestep"))
                intents.append(TapKeyIntent(key=self.jump_key, hold_sec=0.05, reason="tele_jump"))
                return (intents, None, "")

            intents.extend(self._steer_keyboard_toward(world, self.npc_stand, "addon_keep_npc", now))
            return (intents, None, "")

        # Phase 2: Locate 'Addon Keep Teleport' golden sign on screen, then toggle Tab cursor & click NPC
        if self._tele_phase == "FIND_SIGN":
            intents = self.release_all_move_keys("find_tele_sign")
            # NEVER shortcut to teleport_menu before NPC has actually been clicked (_tele_click_attempts > 0)!
            if (
                self._tele_click_attempts > 0
                and world.snapshot.teleport_menu_bbox is not None
                and self._is_valid_popup_near_npc(
                    world.snapshot.teleport_menu_bbox, self._last_tele_npc_click_xy, fw, fh
                )
            ):
                self._tele_phase = "WAIT_MENU"
                self._tele_phase_ts = now
                return (intents, None, "")

            sign = world.snapshot.addon_keep_sign_bbox
            if sign is not None:
                self._sign_missing_since = None
                sx, sy, sw, sh = sign
                sign_cx = sx + sw // 2
                sign_cy = sy + sh // 2
                # Ensure sign is comfortably inside clickable center band before clicking
                if (fw * 0.14) <= sign_cx <= (fw * 0.86):
                    if not self.tracker.is_ready("tele_npc_click", now):
                        return (intents, None, "")
                    # Step A: unlock free cursor via Tab ONLY (never Esc!), wait 0.18s so RYL doesn't spin camera on mouse move!
                    next_attempt = self._tele_click_attempts + 1
                    toggle_key = self.cursor_toggle_key
                    if not self._cursor_toggled_by_us:
                        self._active_cursor_key = toggle_key
                        self._cursor_toggled_by_us = True
                        self._cursor_ready_at = now + 0.20
                        intents.append(
                            TapKeyIntent(key=toggle_key, hold_sec=0.05, reason=f"show_cursor_{toggle_key}")
                        )
                        return (intents, None, "")
                    if now < self._cursor_ready_at:
                        return (intents, None, "")

                    # Step B: cursor is visible -> click NPC torso right beneath 'Addon Keep Teleport' sign!
                    self._tele_click_attempts = next_attempt
                    click_x = max(20, min(fw - 20, sign_cx))
                    click_y = max(20, min(fh - 40, int(round(sign_cy + self.npc_body_offset_y_px * scale_y))))
                    self._last_tele_npc_click_xy = (click_x, click_y)
                    self.tracker.set_cooldown("tele_npc_click", now, 1.8)
                    intents.append(
                        ClickScreenIntent(
                            x=click_x,
                            y=click_y,
                            button="left",
                            hold_sec=self.npc_click_hold_sec,
                            reason=f"click_addon_keep_npc#{self._tele_click_attempts}",
                            no_jitter=True,
                        )
                    )
                    self._tele_phase = "WAIT_MENU"
                    self._tele_phase_ts = now
                    return (intents, None, "")

            # Sign not visible: debounce 1.2s before hiding Tab cursor to scan
            if self._sign_missing_since is None:
                self._sign_missing_since = now
            if self._cursor_toggled_by_us:
                if (now - self._sign_missing_since) < 1.2:
                    return (intents, None, "")
                intents.append(
                    TapKeyIntent(key=self._active_cursor_key, hold_sec=0.04, reason="hide_cursor_to_scan_npc")
                )
                self._cursor_toggled_by_us = False
            if self.tracker.is_ready("tele_scan_turn", now):
                self.tracker.set_cooldown("tele_scan_turn", now, 0.24)
                dx = 175
                intents.append(
                    MoveMouseRelIntent(
                        dx=dx,
                        reason="scan_for_addon_keep_sign",
                        est_yaw_deg=-dx / max(1.0, self.yaw_px_per_deg),
                        steps=5,
                        step_delay=0.002,
                    )
                )
            return (intents, None, "")

        # Phase 3: Wait for 'Talk / Teleport' menu popup and click 'Teleport' (bottom half of popup)
        if self._tele_phase == "WAIT_MENU":
            intents = self.release_all_move_keys("wait_tele_menu")
            menu = world.snapshot.teleport_menu_bbox
            if menu is not None and self._is_valid_popup_near_npc(menu, self._last_tele_npc_click_xy, fw, fh):
                mx, my, mw, mh = menu
                tele_x = mx + mw // 2
                # In teleport_menu.png (43px tall): 'Talk' sits on rows 9..16 (0.21..0.37),
                # 'Teleport' sits on rows 27..34 (0.63..0.79) -> 0.76 hits dead-center of 'Teleport' with zero jitter!
                tele_y = my + int(round(mh * 0.76))
                self.tracker.accept_teleport_jump()
                self._tele_phase = "WAIT_LOAD"
                self._tele_phase_ts = now
                intents.append(
                    ClickScreenIntent(
                        x=tele_x,
                        y=tele_y,
                        button="left",
                        hold_sec=self.npc_click_hold_sec,
                        reason="click_teleport_option",
                        no_jitter=True,
                    )
                )
                logger.info("Clicked 'Teleport' menu option at (%d, %d); waiting for Addon Keep load...", tele_x, tele_y)
                return (intents, None, "")

            # Timeout waiting for popup -> if >=2 failed clicks and dist > 4.5u, walk closer to (1062, 2627)!
            if (now - self._tele_phase_ts) >= 2.2:
                if (
                    coords is not None
                    and (
                        (self._tele_click_attempts >= 2 and coords.distance_to(self.npc_stand) > 4.5)
                        or coords.distance_to(self.npc_stand) > (self.npc_arrival_radius + 4.0)
                    )
                ):
                    if self._cursor_toggled_by_us:
                        intents.append(
                            TapKeyIntent(key=self._active_cursor_key, hold_sec=0.04, reason="hide_cursor_reapproach_npc")
                        )
                        self._cursor_toggled_by_us = False
                    self._tele_click_attempts = 0
                    self._tele_phase = "WALK_TO_NPC"
                else:
                    self._tele_phase = "FIND_SIGN"
                self._tele_phase_ts = now
            return (intents, None, "")

        # Phase 4: Wait for 'Connecting to the server...' (~5-6s) until landing at Addon Keep
        if self._tele_phase == "WAIT_LOAD":
            self.tracker.accept_teleport_jump()
            if (now - self._tele_phase_ts) >= self.teleport_load_timeout_sec:
                logger.warning("Teleport load timed out after %.1fs; retrying Addon Keep NPC click", self.teleport_load_timeout_sec)
                self._tele_phase = "FIND_SIGN"
                self._tele_phase_ts = now
            return (self.release_all_move_keys("wait_tele_loading"), None, "")

        return ([], None, "")

    def _build_arrival_intents(self, coords: Point2D, now: float, reason: str) -> list[ActionIntent]:
        """Release travel keys cleanly upon arriving at farming spot (never toggle Mouse Mode via Esc)."""
        intents = self.release_all_move_keys(reason)
        self._mode_switched_to_kb = False
        logger.info(
            "Arrived at farming spot (%.0f, %.0f) -> resuming farming!",
            coords.x,
            coords.y,
        )
        return intents

    def step_travel_to_spot(
        self,
        world: WorldState,
        now: float,
        travel_waypoints: list[Waypoint],
        spot_center: Point2D,
        farm_radius: float,
    ) -> tuple[list[ActionIntent], BotState | None, str]:
        """Run smoothly along travel_waypoints from Ardun Keep (1409,1773) -> farming spot in 100% Keyboard Mode (WASD)."""
        coords = world.coords_smoothed
        if coords is None:
            return ([], None, "")

        # Only declare arrival once SOLIDLY inside farm_radius (<= 0.78 * farm_radius) so SEARCH_MOB
        # resumes farming cleanly without bouncing back to NAVIGATE_TO_SPOT!
        spot_entry_r = min(farm_radius * 0.78, max(self.travel_final_radius, 15.0))
        if coords.distance_to(spot_center) <= spot_entry_r:
            self.reset_travel_state()
            return (
                self._build_arrival_intents(coords, now, "arrived_farming_spot"),
                BotState.SEARCH_MOB,
                "arrived_at_farm_spot",
            )

        # Ensure the waypoint sequence always finishes solidly at spot_center
        wps = list(travel_waypoints) if travel_waypoints else []
        if not wps or Point2D(wps[-1].x, wps[-1].y).distance_to(spot_center) > (spot_entry_r * 0.5):
            wps.append(Waypoint(name="spot_center", x=spot_center.x, y=spot_center.y))

        # Advance through any waypoints already passed
        idx = max(0, min(self.tracker.travel_waypoint_index, len(wps) - 1))
        while idx < len(wps):
            wp = wps[idx]
            is_last = idx == len(wps) - 1
            rad = spot_entry_r if is_last else self.travel_arrival_radius
            if coords.distance_to(Point2D(wp.x, wp.y)) <= rad:
                idx += 1
                self.tracker.travel_waypoint_index = idx
                self._travel_unstuck_count = 0
                if is_last:
                    self.reset_travel_state()
                    return (
                        self._build_arrival_intents(coords, now, "arrived_last_travel_wp"),
                        BotState.SEARCH_MOB,
                        "arrived_at_farm_spot",
                    )
            else:
                break

        idx = min(idx, len(wps) - 1)
        self.tracker.travel_waypoint_index = idx
        active_wp = wps[idx]

        # Fast diagonal boulder/tree/riverbank sidestep when stuck during travel (100% Keyboard Mode WASD + Space)
        if now < self._travel_unstuck_until:
            return (
                [
                    HoldKeyIntent(key=self.forward_key, hold=True, reason="travel_unstuck_w", move_rel_angle_deg=0.0),
                    HoldKeyIntent(
                        key=self._travel_unstuck_dir,
                        hold=True,
                        reason=f"travel_unstuck_{self._travel_unstuck_dir}",
                    ),
                ],
                None,
                "",
            )

        if world.is_stuck:
            self._travel_unstuck_count += 1
            self._travel_unstuck_dir = (
                self.strafe_left_key if (self._travel_unstuck_count % 2 == 1) else self.strafe_right_key
            )
            self._travel_unstuck_until = now + 0.65
            self.tracker.clear_stuck(reset_consecutive=True)
            turn_dx = -120 if self._travel_unstuck_dir == self.strafe_left_key else 120
            logger.info(
                "Travel obstacle sidestep #%d (%s + jump) toward %s",
                self._travel_unstuck_count,
                self._travel_unstuck_dir.upper(),
                active_wp.name,
            )
            return (
                [
                    HoldKeyIntent(key=self.forward_key, hold=True, reason="travel_unstuck_start", move_rel_angle_deg=0.0),
                    HoldKeyIntent(key=self._travel_unstuck_dir, hold=True, reason="travel_unstuck_strafe"),
                    TapKeyIntent(key=self.jump_key, hold_sec=0.05, reason="travel_unstuck_jump"),
                    MoveMouseRelIntent(
                        dx=turn_dx,
                        reason="travel_unstuck_yaw",
                        est_yaw_deg=-turn_dx / max(1.0, self.yaw_px_per_deg),
                        steps=4,
                        step_delay=0.003,
                    ),
                ],
                None,
                "",
            )

        # Ensure strafe keys are released after unstuck finishes
        intents: list[ActionIntent] = [
            HoldKeyIntent(key=self.back_key, hold=False, reason="travel_rel_s"),
            HoldKeyIntent(key=self.strafe_left_key, hold=False, reason="travel_rel_a"),
            HoldKeyIntent(key=self.strafe_right_key, hold=False, reason="travel_rel_d"),
        ]
        intents.extend(self._steer_keyboard_toward(world, Point2D(active_wp.x, active_wp.y), active_wp.name, now))
        return (intents, None, "")
