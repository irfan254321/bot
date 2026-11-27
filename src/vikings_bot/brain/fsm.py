"""Mouse-Mode Farming, Death Respawn, Return-to-Spot & 80m Auto-Repair Finite State Machine."""

from __future__ import annotations

import logging
import math

from vikings_bot.brain.mouse_farm import MouseFarmController
from vikings_bot.brain.navigation import MouseModeNavigator
from vikings_bot.brain.respawn_travel import RespawnTravelController
from vikings_bot.config import BotConfig
from vikings_bot.input.actions import (
    ActionIntent,
    ClickScreenIntent,
    DragCameraIntent,
    HoldKeyIntent,
    TapKeyIntent,
)
from vikings_bot.state.models import BotMode, BotState, MobTarget, Point2D, WorldState
from vikings_bot.state.tracker import WorldStateTracker

logger = logging.getLogger(__name__)


class FarmingFSM:
    """Orchestrates Auto-ESC Mouse Guard -> Search -> 1x Click Combat Wait -> Loot -> MMB Sweep -> 80m Repair -> Respawn."""

    def __init__(
        self,
        config: BotConfig,
        tracker: WorldStateTracker,
        mode: BotMode = BotMode.FULL,
    ) -> None:
        self.config = config
        self.tracker = tracker
        self.mode = mode
        self.navigator = MouseModeNavigator(config, tracker)
        self.farm = MouseFarmController(config, tracker)
        self.respawn_travel = RespawnTravelController(config, tracker)

        f = config.farming
        c = config.controls
        p = config.perception

        self.recovery_duration_sec = float(f.get("recovery_duration_sec", 1.4))
        self.anchor_spot_on_resume = bool(f.get("anchor_spot_on_resume", False))
        self.fallback_route_name: str | None = str(f.get("fallback_route_name") or "").strip() or None
        self.fallback_after_deaths: int = max(0, int(f.get("fallback_after_deaths", 0)))
        self.max_sweeps_before_patrol: int = max(1, int(f.get("max_sweeps_before_patrol", 4)))
        self.combat_miss_timeout_sec: float = float(f.get("combat_miss_timeout_sec", 2.4))
        self.combat_min_lock_hold_sec: float = max(0.0, float(f.get("combat_min_lock_hold_sec", 2.0)))
        self.combat_no_hit_abort_sec: float = float(f.get("combat_no_hit_abort_sec", 4.5))
        self.combat_hit_confirm_timeout_sec: float = float(
            f.get("combat_hit_confirm_timeout_sec", self.combat_no_hit_abort_sec)
        )
        self.combat_confirmed_max_sec: float = float(f.get("combat_confirmed_max_sec", 35.0))
        self.combat_post_last_hit_idle_sec: float = float(
            f.get("combat_post_last_hit_idle_sec", 2.5)
        )
        self.spot_center_x: float = float(f.get("spot_x", 627.0))
        self.spot_center_y: float = float(f.get("spot_y", 1817.0))
        self.spot_center_arrival_radius: float = max(2.0, float(f.get("spot_center_arrival_radius", 12.0)))
        self.return_center_min_drift_units: float = max(
            self.spot_center_arrival_radius, float(f.get("return_center_min_drift_units", 35.0))
        )
        self.return_center_timeout_sec: float = max(1.0, float(f.get("return_center_timeout_sec", 6.0)))
        self.loot_dead_point_sweep: bool = bool(f.get("loot_dead_point_sweep", True))
        self.loot_click_hold_sec: float = max(0.04, float(f.get("loot_click_hold_sec", 0.10)))
        self.loot_click_pause_e_sec: float = max(0.0, float(f.get("loot_click_pause_e_sec", 0.12)))
        self.combat_ghost_lock_sec: float = float(f.get("combat_ghost_lock_sec", 1.5))
        self.combat_ghost_blacklist_sec: float = float(f.get("combat_ghost_blacklist_sec", 15.0))
        self.combat_blind_timeout_sec: float = float(f.get("combat_blind_timeout_sec", 10.0))
        self.combat_stagnant_hp_sec: float = float(f.get("combat_stagnant_hp_sec", 12.0))
        self.combat_hard_timeout_sec: float = float(f.get("combat_hard_timeout_sec", 28.0))
        self.combat_zero_drop_kill_sec: float = float(f.get("combat_zero_drop_kill_sec", 1.6))
        self.loot_clear_confirm_frames: int = max(2, int(f.get("loot_clear_confirm_frames", 4)))
        self.loot_max_attempts_per_bag: int = max(1, int(f.get("loot_max_attempts_per_bag", 2)))
        self.loot_attempt_grid_px: int = max(8, int(f.get("loot_attempt_grid_px", 24)))
        self.loot_visited_grid_ttl_sec: float = max(2.0, float(f.get("loot_visited_grid_ttl_sec", 15.0)))
        self.mandatory_loot_duration_sec: float = max(1.0, float(f.get("mandatory_loot_duration_sec", 2.5)))
        self.post_kill_loot_window_sec: float = max(
            self.mandatory_loot_duration_sec, float(f.get("post_kill_loot_window_sec", 7.5))
        )
        self.loot_min_window_sec: float = max(0.5, float(f.get("loot_min_window_sec", 2.5)))
        self.loot_empty_exit_sec: float = max(0.2, float(f.get("loot_empty_exit_sec", 0.8)))
        self.loot_target_move_timeout_sec: float = max(0.4, float(f.get("loot_target_move_timeout_sec", 1.2)))
        self.loot_fallback_step_interval_sec: float = max(
            0.25, float(f.get("loot_fallback_step_interval_sec", 0.75))
        )
        self.loot_fallback_sweep_offset_px: int = max(20, int(f.get("loot_fallback_sweep_offset_px", 85)))
        self.loot_max_total_sec: float = float(f.get("loot_max_total_sec", self.post_kill_loot_window_sec))
        self.last_kill_loot_radius_px: float = float(p.get("last_kill_loot_radius_px", 140.0))
        self.last_kill_loot_window_sec: float = float(p.get("last_kill_loot_window_sec", 0.80))
        self.anti_ks_enabled: bool = bool(p.get("anti_ks_enabled", True))
        self.anti_ks_player_radius_px: float = float(p.get("anti_ks_player_radius_px", 240.0))
        self.anti_ks_damage_zone_radius_px: float = float(p.get("anti_ks_damage_zone_radius_px", 220.0))
        self.anti_ks_damage_zone_ttl_sec: float = float(p.get("anti_ks_damage_zone_ttl_sec", 6.0))
        self.anti_ks_min_untouched_hp_ratio: float = float(p.get("anti_ks_min_untouched_hp_ratio", 0.94))

        self.control_mode_setting: str = str(c.get("control_mode", "mouse")).lower()
        self.control_mode: str = str(c.get("mode", "mouse")).lower()
        self.forward_key: str = str(c.get("move_forward_key", "w"))
        self.back_key: str = str(c.get("move_back_key", "s"))
        self.strafe_left_key: str = str(c.get("strafe_left_key", "a"))
        self.strafe_right_key: str = str(c.get("strafe_right_key", "d"))
        self.hp_potion_key = c.get("hp_potion_key", "0")
        self.mp_potion_key = c.get("mp_potion_key", "9")
        self.hp_threshold = float(c.get("hp_potion_threshold", c.get("hp_threshold", 0.65)))
        self.mp_threshold = float(c.get("mp_potion_threshold", c.get("mp_threshold", 0.30)))
        self.hp_cd = float(c.get("hp_potion_cooldown_sec", 1.5))
        self.mp_cd = float(c.get("mp_potion_cooldown_sec", 2.0))
        self.weapon_swap_key = str(c.get("weapon_swap_key", "q"))
        self.weapon_min_ea = int(c.get("weapon_min_ea", 130))
        self.weapon_swap_cd = float(c.get("weapon_swap_cooldown_sec", 2.5))
        self.auto_restore_mouse_mode: bool = (
            bool(c.get("auto_restore_mouse_mode", True)) and self.control_mode == "mouse"
        )

        self.state: BotState = BotState.PAUSED
        self._need_dynamic_anchor: bool = False
        self._just_resumed: bool = False
        self._state_entered_at: float = 0.0
        self._recovery_attempts: int = 0
        self.kills_count: int = 0
        self.deaths_count: int = 0
        self.repairs_count: int = 0
        self._active_farming_sec: float = 0.0
        self._last_step_ts: float | None = None
        self._repair_weapon_checked: bool = False
        self._force_repair_requested: bool = False
        self._fallback_route_activated: bool = False
        self._was_outside_city: bool = False
        self._last_false_death_log_ts: float = -999.0

        # One-shot startup ENSURE_MOUSE_MODE / ENSURE_KEYBOARD_MODE state
        self._ensure_mode_enabled: bool = self.auto_restore_mouse_mode
        self._mouse_mode_verified: bool = False
        self._ensure_mouse_phase: str = "SEND_FIRST_ESC"
        self._ensure_mouse_until: float = 0.0
        self._post_ensure_state: BotState = BotState.SEARCH_MOB
        self._ensure_kb_phase: str = "SEND_FIRST_ESC"
        self._ensure_kb_until: float = 0.0
        self._post_ensure_kb_state: BotState = BotState.TELEPORT_TO_SPOT
        self._mouse_restore_wait_until: float = 0.0
        self._post_drag_settle_until: float = 0.0
        self._sweeps_in_search: int = 0

        # COMBAT_WAIT (1-click, hit-confirmed combat, mandatory loot) tracking
        self.active_lock: MobTarget | None = None
        self._combat_click_xy: tuple[int, int] | None = None
        self._combat_confirmed: bool = False
        self._combat_confirmed_at: float | None = None
        self._combat_last_hit_time: float | None = None
        self._combat_started_fresh: bool = True
        self._combat_first_hp_checked: bool = False
        self._combat_saw_hp: bool = False
        self._combat_last_hp_ts: float | None = None
        self._combat_min_hp: float = 1.0
        self._combat_last_hp_drop_ts: float = 0.0
        self._combat_anchor_pos: Point2D | None = None
        self._combat_last_move_ts: float = 0.0
        self._combat_min_dist_px: float = 9999.0
        self._combat_last_approach_ts: float = 0.0
        self._combat_loot_streak: int = 0
        self._aborted_mobs: list[tuple[float, float, float]] = []  # (expire_ts, cx, cy)
        self.external_combat_zones: list[tuple[float, float, float]] = []  # (expire_ts, x, y)

        # LOOT_PICKUP sequential visited_loot_grids on 24x24px grid (TTL 15.0s, max 2 attempts / 1.2s move timeout)
        self.visited_loot_grids: dict[tuple[int, int], tuple[int, float]] = {}
        self._loot_grid_attempts: dict[tuple[int, int], int] = {}
        self._loot_active_grid: tuple[int, int] | None = None
        self._loot_active_grid_since: float = 0.0
        self._loot_fallback_stage: int = 0
        self._loot_fallback_next_ts: float = 0.0
        self._loot_saw_any_bag: bool = False
        self._loot_prev_count: int = 0
        self._loot_clear_streak: int = 0
        self._loot_empty_since: float | None = None
        self.last_blacklisted_loot: tuple[float, float] | None = None

    def set_control_mode(self, mode: str) -> None:
        m = str(mode).strip().lower()
        if m in ("mouse", "keyboard"):
            self.control_mode = m
            self.navigator.control_mode = m
            self.respawn_travel.control_mode = m
            self.auto_restore_mouse_mode = (
                bool(self.config.controls.get("auto_restore_mouse_mode", True)) and m == "mouse"
            )

    def release_move_keys(self, reason: str) -> list[ActionIntent]:
        self.tracker.notify_movement_stopped()
        return [
            HoldKeyIntent(key=self.forward_key, hold=False, reason=reason),
            HoldKeyIntent(key=self.back_key, hold=False, reason=reason),
            HoldKeyIntent(key=self.strafe_left_key, hold=False, reason=reason),
            HoldKeyIntent(key=self.strafe_right_key, hold=False, reason=reason),
        ]

    def _check_survival_potions(self, state: WorldState, now: float) -> list[ActionIntent]:
        intents: list[ActionIntent] = []
        if (
            self.hp_potion_key
            and 0.01 < state.hp_ratio < self.hp_threshold
            and self.tracker.is_ready("hp_potion", now)
        ):
            self.tracker.set_cooldown("hp_potion", now, self.hp_cd)
            intents.append(TapKeyIntent(key=str(self.hp_potion_key), reason=f"hp_pot({state.hp_ratio:.0%})"))

        if (
            self.mp_potion_key
            and state.mp_ratio < self.mp_threshold
            and self.tracker.is_ready("mp_potion", now)
        ):
            self.tracker.set_cooldown("mp_potion", now, self.mp_cd)
            intents.append(TapKeyIntent(key=str(self.mp_potion_key), reason=f"mp_pot({state.mp_ratio:.0%})"))
        return intents

    def _check_weapon_swap(self, state: WorldState, now: float) -> list[ActionIntent]:
        ea = state.snapshot.ea_value
        if ea is None or ea >= self.weapon_min_ea or not self.weapon_swap_key:
            return []
        if not self.tracker.is_ready("weapon_swap", now):
            return []
        self.tracker.set_cooldown("weapon_swap", now, self.weapon_swap_cd)
        return [TapKeyIntent(key=self.weapon_swap_key, reason=f"weapon_swap_ea({ea}<{self.weapon_min_ea})")]

    def _record_aborted_mob(
        self,
        mob: MobTarget | None,
        now: float,
        ttl_sec: float = 6.0,
        click_xy: tuple[int, int] | None = None,
    ) -> None:
        self._aborted_mobs = [m for m in self._aborted_mobs if now < m[0]]
        if mob is not None:
            self._aborted_mobs.append((now + ttl_sec, float(mob.click_x), float(mob.click_y)))
        elif click_xy is not None:
            self._aborted_mobs.append((now + ttl_sec, float(click_xy[0]), float(click_xy[1])))

    def _is_aborted_mob(self, mob: MobTarget, now: float, scale_x: float) -> bool:
        if not self._aborted_mobs:
            return False
        self._aborted_mobs = [m for m in self._aborted_mobs if now < m[0]]
        lim = 105.0 * max(0.55, scale_x)
        return any(math.hypot(mob.click_x - ax, mob.click_y - ay) <= lim for _exp, ax, ay in self._aborted_mobs)

    def _update_external_combat_zones(self, world: WorldState, now: float) -> None:
        """Record external floating damage points observed while NOT in COMBAT_WAIT (TTL 6.0s)."""
        self.external_combat_zones = [z for z in self.external_combat_zones if now < z[0]]
        if not self.anti_ks_enabled or not world.external_damage_points:
            return
        exp = now + self.anti_ks_damage_zone_ttl_sec
        for dx, dy in world.external_damage_points:
            self.external_combat_zones.append((exp, float(dx), float(dy)))

    def _is_in_anti_ks_zone(
        self,
        x: float,
        y: float,
        world: WorldState,
        now: float,
        *,
        player_radius_px: float | None = None,
        damage_radius_px: float | None = None,
    ) -> bool:
        """Return True if (x, y) is too close to another player or an external combat damage zone."""
        if not self.anti_ks_enabled:
            return False
        self.external_combat_zones = [z for z in self.external_combat_zones if now < z[0]]
        fw = max(1, world.snapshot.frame_width)
        sx_1024 = 1.0 if fw <= 1100 else (fw / 1024.0)
        p_rad = (self.anti_ks_player_radius_px if player_radius_px is None else player_radius_px) * sx_1024
        d_rad = (self.anti_ks_damage_zone_radius_px if damage_radius_px is None else damage_radius_px) * sx_1024
        if any(math.hypot(x - opx, y - opy) <= p_rad for opx, opy in world.snapshot.other_players):
            return True
        if any(math.hypot(x - zx, y - zy) <= d_rad for _exp, zx, zy in self.external_combat_zones):
            return True
        return False

    def transition(self, new_state: BotState, now: float, reason: str = "") -> None:
        if new_state == self.state:
            return
        old = self.state
        self.state = new_state
        self._state_entered_at = now
        if new_state in (BotState.PAUSED, BotState.EMERGENCY_STOP):
            self.tracker.clear_target_lock()
            self.tracker.notify_movement_stopped()
        elif new_state in (BotState.SEARCH_MOB, BotState.AT_SPOT_IDLE):
            self._sweeps_in_search = 0
            self.farm.reset_sweep_direction()
            self.tracker.clear_target_lock()
            self.tracker.notify_movement_stopped()
        elif new_state == BotState.COMBAT_WAIT:
            self._combat_loot_streak = 0
            self._combat_first_hp_checked = False
            self.tracker.notify_movement_stopped()
        elif new_state == BotState.LOOT:
            self.visited_loot_grids.clear()
            self._loot_grid_attempts.clear()
            self._loot_active_grid = None
            self._loot_active_grid_since = now
            self._loot_fallback_stage = 0
            self._loot_fallback_next_ts = now
            self._loot_saw_any_bag = False
            self._loot_prev_count = 0
            self._loot_clear_streak = 0
            self._loot_empty_since = None
            self.tracker.clear_target_lock()
            self.tracker.notify_movement_stopped()
        elif new_state == BotState.DEAD_RESPAWN:
            self.tracker.reset_for_respawn()
            self.respawn_travel.reset_teleport_state(now)
            self.respawn_travel.reset_travel_state()
            self.farm.request_daylight_weather_sync()
        elif new_state == BotState.REPAIR_RECALL:
            self.tracker.clear_target_lock()
            self.tracker.notify_movement_stopped()
            self.respawn_travel.reset_repair_recall_state(now)
        elif new_state == BotState.REPAIR_AT_NPC:
            self.tracker.clear_target_lock()
            self.tracker.notify_movement_stopped()
            self.respawn_travel.reset_repair_npc_state(now)
            self._mouse_mode_verified = False
        elif new_state == BotState.TELEPORT_TO_SPOT:
            self.tracker.clear_target_lock()
            self.tracker.notify_movement_stopped()
            self.respawn_travel.reset_teleport_state(now)
            self._mouse_mode_verified = False
        elif new_state == BotState.TRAVEL_TO_SPOT:
            self.tracker.clear_target_lock()
            self.tracker.notify_movement_stopped()
            self.respawn_travel.reset_travel_state()
            self.farm.request_daylight_weather_sync()
            self._mouse_mode_verified = False
        elif new_state == BotState.RECOVER_STUCK:
            self._recovery_attempts += 1
            self.tracker.clear_target_lock()
        logger.info("FSM %s -> %s (%s)", old.value, new_state.value, reason)

    def _enter_ensure_mouse_mode(self, next_state: BotState, now: float, reason: str) -> None:
        self._mouse_mode_verified = False
        self._ensure_mouse_phase = "SEND_FIRST_ESC"
        self._ensure_mouse_until = 0.0
        self._post_ensure_state = next_state
        self.transition(BotState.ENSURE_MOUSE_MODE, now, reason)

    def _enter_ensure_keyboard_mode(self, next_state: BotState, now: float, reason: str) -> None:
        self._mouse_mode_verified = False
        self._ensure_kb_phase = "SEND_FIRST_ESC"
        self._ensure_kb_until = 0.0
        self._post_ensure_kb_state = next_state
        self.transition(BotState.ENSURE_KEYBOARD_MODE, now, reason)

    def resume(self, now: float, *, ensure_mouse_mode: bool | None = None) -> None:
        if self.state == BotState.EMERGENCY_STOP:
            return
        self._just_resumed = True
        self._recovery_attempts = 0
        self.tracker.reset_for_resume()
        self.farm.request_daylight_weather_sync()

        if self.mode == BotMode.NAVIGATE:
            target_state = BotState.NAVIGATE_TO_SPOT
            rsn = "resume_navigate"
        elif self.mode == BotMode.LOOT:
            target_state = BotState.LOOT
            rsn = "resume_loot"
        elif self.mode == BotMode.COMBAT:
            target_state = BotState.SEARCH_MOB
            rsn = "resume_combat"
        else:
            self._need_dynamic_anchor = self.anchor_spot_on_resume
            target_state = BotState.SEARCH_MOB
            rsn = "resume_full"

        want_ensure = self.auto_restore_mouse_mode if ensure_mouse_mode is None else bool(ensure_mouse_mode)
        self._ensure_mode_enabled = want_ensure
        if want_ensure and self.mode in (BotMode.FULL, BotMode.COMBAT, BotMode.LOOT):
            self._enter_ensure_mouse_mode(target_state, now, f"{rsn}_ensure_mouse_mode")
            return

        self._mouse_mode_verified = True
        self.transition(target_state, now, rsn)

    def pause(self, now: float, reason: str = "manual_pause") -> None:
        if self.state != BotState.EMERGENCY_STOP:
            self._just_resumed = False
            self._last_step_ts = None
            self.transition(BotState.PAUSED, now, reason)

    def request_repair_now(self) -> None:
        self._force_repair_requested = True

    @property
    def repair_remaining_minutes(self) -> float | None:
        if not self.respawn_travel.repair_enabled:
            return None
        if self._force_repair_requested:
            return 0.0
        return max(0.0, (self.respawn_travel.repair_interval_sec - self._active_farming_sec) / 60.0)

    def emergency_stop(self, now: float, reason: str = "kill_switch") -> None:
        self._just_resumed = False
        self.transition(BotState.EMERGENCY_STOP, now, reason)

    @property
    def configured_spot_center(self) -> Point2D:
        if self.config.active_route_name == "gold_rustler_spot":
            return Point2D(self.spot_center_x, self.spot_center_y)
        if self.navigator.route.waypoints:
            w0 = self.navigator.route.waypoints[0]
            return Point2D(w0.x, w0.y)
        return self.navigator.spot_center

    @property
    def travel_status_desc(self) -> str:
        if self.state == BotState.DEAD_RESPAWN:
            return f"RESPAWN #{self.deaths_count}"
        if self.state == BotState.REPAIR_RECALL:
            return "RECALL /kill"
        if self.state == BotState.REPAIR_AT_NPC:
            return f"REPAIR:{self.respawn_travel._repair_phase}"
        if self.state == BotState.TELEPORT_TO_SPOT:
            return f"TELE:{self.respawn_travel._tele_phase}"
        if self.state == BotState.TRAVEL_TO_SPOT:
            total = max(1, len(self.navigator.route.travel_waypoints))
            cur = min(total, self.tracker.travel_waypoint_index + 1)
            return f"WP {cur}/{total}"
        return ""

    def _activate_fallback_route_if_needed(self) -> None:
        if (
            self._fallback_route_activated
            or not self.fallback_route_name
            or self.fallback_after_deaths <= 0
            or self.deaths_count < self.fallback_after_deaths
            or self.fallback_route_name not in self.config.routes
        ):
            return
        self._fallback_route_activated = True
        old_route = self.config.active_route_name
        self.config.active_route_name = self.fallback_route_name
        self.navigator.route = self.config.active_route
        self.navigator.set_dynamic_anchor(None)
        self.tracker.waypoint_index = 0
        self.tracker.travel_waypoint_index = 0
        fb_c = self.configured_spot_center
        logger.warning(
            "[SPOT FALLBACK] Mati %dx di spot utama -> pindah otomatis dari %s ke %s (%.0f, %.0f)!",
            self.deaths_count,
            old_route,
            self.fallback_route_name,
            fb_c.x,
            fb_c.y,
        )

    def _is_confirmed_real_death(self, world: WorldState, now: float) -> tuple[bool, str]:
        """Anti-False-Death Guard: require >=5 consecutive 0%-HP frames AND corroborating evidence.

        Prevents HUD glitches / 1024x768 ROI misreads (HP dropping suddenly from >30% to 0%)
        from triggering false DEAD_RESPAWN transitions while still alive at the farming spot.
        """
        min_frames = max(5, int(self.tracker.death_zero_hp_frames))
        if world.zero_hp_streak < min_frames:
            return (False, "streak_below_min")

        if world.respawn_dialog_visible:
            return (True, "respawn_dialog")
        if world.hp_dropped_below_35_recently or world.last_nonzero_hp_ratio < 0.35:
            return (True, "prior_low_hp")
        if self._was_outside_city and self.respawn_travel.is_in_city(world.coords_smoothed):
            return (True, "warped_to_city")

        if (now - self._last_false_death_log_ts) >= 5.0:
            self._last_false_death_log_ts = now
            logger.warning(
                "Anti-False-Death: suppressed DEAD_RESPAWN (hp=0%% streak=%d, prev_hp=%.0f%%, respawn_ui=%s)",
                world.zero_hp_streak,
                world.last_nonzero_hp_ratio * 100.0,
                world.respawn_dialog_visible,
            )
        return (False, "suppressed_false_death")

    def _check_periodic_repair_due(self, world: WorldState, now: float) -> list[ActionIntent] | None:
        if self.mode != BotMode.FULL or world.pending_loot:
            return None
        due = self._force_repair_requested or (
            self.respawn_travel.repair_enabled and self._active_farming_sec >= self.respawn_travel.repair_interval_sec
        )
        if not due:
            return None
        if (
            not self._repair_weapon_checked
            and world.snapshot.ea_value is not None
            and world.snapshot.ea_value < self.weapon_min_ea
        ):
            self._repair_weapon_checked = True
            return self._check_weapon_swap(world, now)
        rsn = (
            "manual_repair_now"
            if self._force_repair_requested
            else f"periodic_repair_{self.respawn_travel.repair_interval_sec / 60.0:.0f}m"
        )
        self._force_repair_requested = False
        self._need_dynamic_anchor = False
        if self.respawn_travel.is_in_city(world.coords_smoothed) or self.state == BotState.TELEPORT_TO_SPOT:
            if self._ensure_mode_enabled and self.state != BotState.TELEPORT_TO_SPOT:
                self._enter_ensure_keyboard_mode(BotState.REPAIR_AT_NPC, now, f"{rsn}_already_in_town_ensure_kb")
            else:
                self.transition(BotState.REPAIR_AT_NPC, now, f"{rsn}_already_in_town")
            return self.respawn_travel.release_all_move_keys("repair_in_town_start")
        self.transition(BotState.REPAIR_RECALL, now, rsn)
        return self.respawn_travel.release_all_move_keys("enter_repair_recall")

    def _enter_combat_wait(self, world: WorldState, mob: MobTarget, now: float, reason: str) -> list[ActionIntent]:
        lid = self.tracker.lock_target(mob, now)
        world.locked_mob = mob
        world.locked_mob_id = lid
        world.locked_mob_visible = True
        world.locked_mob_lost = False
        self.transition(BotState.COMBAT_WAIT, now, f"{reason}[lock#{lid}]")
        self._combat_click_xy = (int(mob.click_x), int(mob.click_y))
        self._combat_confirmed = False
        self._combat_confirmed_at = None
        self._combat_last_hit_time = None
        self._combat_started_fresh = not bool(mob.has_hp_bar)
        self._combat_first_hp_checked = False
        self._combat_saw_hp = bool(mob.has_hp_bar)
        self._combat_last_hp_ts = now if mob.has_hp_bar else None
        self._combat_min_hp = float(mob.hp_ratio_est) if mob.hp_ratio_est is not None else 1.0
        self._combat_last_hp_drop_ts = now
        self._combat_anchor_pos = world.coords_smoothed
        self._combat_last_move_ts = now
        self._combat_min_dist_px = float(mob.dist_to_player_px)
        self._combat_last_approach_ts = now
        self._combat_loot_streak = 0
        return self.farm.build_engage_click(mob)

    def step(self, world: WorldState) -> list[ActionIntent]:
        now = world.snapshot.timestamp
        self.last_blacklisted_loot = None
        if self.state in (BotState.PAUSED, BotState.EMERGENCY_STOP):
            self._last_step_ts = None
            return []

        if self._last_step_ts is not None and self.state not in (
            BotState.ENSURE_MOUSE_MODE,
            BotState.ENSURE_KEYBOARD_MODE,
            BotState.REPAIR_RECALL,
            BotState.REPAIR_AT_NPC,
        ):
            dt = min(1.0, max(0.0, now - self._last_step_ts))
            self._active_farming_sec += dt
        self._last_step_ts = now

        # --- ONE-SHOT MODE LOCK STATES (Mouse Mode for farming, Keyboard Mode for town/repair/teleport/travel) ---
        if self.state == BotState.ENSURE_MOUSE_MODE:
            return self._handle_ensure_mouse_mode(world, now)
        if self.state == BotState.ENSURE_KEYBOARD_MODE:
            return self._handle_ensure_keyboard_mode(world, now)

        # Track whether character was outside town so sudden city-warp corroborates genuine respawn
        if world.coords_smoothed is not None and not self.respawn_travel.is_in_city(world.coords_smoothed):
            self._was_outside_city = True

        # --- GLOBAL DEATH INTERCEPTION (with Anti-False-Death Guard) ---
        if (
            self.mode == BotMode.FULL
            and self.respawn_travel.enabled
            and self.state not in (BotState.DEAD_RESPAWN, BotState.REPAIR_RECALL)
        ):
            confirmed_dead, death_cue = self._is_confirmed_real_death(world, now)
            if confirmed_dead:
                self.deaths_count += 1
                self._activate_fallback_route_if_needed()
                prev_st = self.state.value
                self._need_dynamic_anchor = False
                self._just_resumed = False
                self._was_outside_city = False
                self.transition(BotState.DEAD_RESPAWN, now, f"died_in_{prev_st}[{death_cue}]#death_{self.deaths_count}")
                return self.respawn_travel.release_all_move_keys("died_release_all")

        # --- SMART RESUME LOCATION CHECK ---
        if self._just_resumed and world.coords_smoothed is not None:
            self._just_resumed = False
            if self.mode == BotMode.FULL and self.respawn_travel.enabled:
                if self.respawn_travel.is_in_city(world.coords_smoothed):
                    self._need_dynamic_anchor = False
                    if self._force_repair_requested or (
                        self.respawn_travel.repair_enabled
                        and self._active_farming_sec >= self.respawn_travel.repair_interval_sec
                    ):
                        self._force_repair_requested = False
                        self.transition(BotState.REPAIR_AT_NPC, now, "resumed_in_town_auto_repair")
                    else:
                        self.transition(BotState.TELEPORT_TO_SPOT, now, "resumed_in_town_auto_teleport")
                elif self.respawn_travel.is_near_addon_keep(world.coords_smoothed):
                    self._need_dynamic_anchor = False
                    self.tracker.travel_waypoint_index = self.respawn_travel.choose_starting_travel_index(
                        world.coords_smoothed, self.navigator.route.travel_waypoints
                    )
                    self.transition(BotState.TRAVEL_TO_SPOT, now, "resumed_at_addon_keep_auto_travel")

        if (
            self._need_dynamic_anchor
            and world.coords_smoothed is not None
            and self.state
            not in (
                BotState.DEAD_RESPAWN,
                BotState.REPAIR_RECALL,
                BotState.REPAIR_AT_NPC,
                BotState.TELEPORT_TO_SPOT,
                BotState.TRAVEL_TO_SPOT,
            )
        ):
            self.navigator.set_dynamic_anchor(world.coords_smoothed)
            self._need_dynamic_anchor = False
            logger.info(
                "Anchored farm spot center to current position (%.0f, %.0f) radius=%.0f",
                world.coords_smoothed.x,
                world.coords_smoothed.y,
                self.navigator.route.farm_radius,
            )

        # --- IMMEDIATE F7 FORCE REPAIR INTERCEPTION (even while in TELEPORT_TO_SPOT or TRAVEL_TO_SPOT!) ---
        if self._force_repair_requested and self.state in (
            BotState.TELEPORT_TO_SPOT,
            BotState.TRAVEL_TO_SPOT,
        ):
            self._force_repair_requested = False
            self._need_dynamic_anchor = False
            if self.respawn_travel.is_in_city(world.coords_smoothed) or self.state == BotState.TELEPORT_TO_SPOT:
                # Already in Green Wales town (and already in Keyboard Mode if in TELEPORT_TO_SPOT) -> go straight to REPAIR_AT_NPC without /kill!
                self.transition(BotState.REPAIR_AT_NPC, now, "manual_repair_now_already_in_town")
                return self.respawn_travel.release_all_move_keys("f7_in_town_repair_start")
            self.transition(BotState.REPAIR_RECALL, now, "manual_repair_now_from_travel")
            return self.respawn_travel.release_all_move_keys("f7_from_travel_recall")

        if self.state == BotState.COMBAT_WAIT:
            # Priority #1 Hard-Gate at top of fsm.step(): nothing can interrupt COMBAT_WAIT
            # except confirmed death; force zero mouse/camera/ESC actions!
            self.active_lock = None
            elapsed = max(0.0, now - self._state_entered_at)
            intents = self._check_survival_potions(world, now)
            intents.extend(self._check_weapon_swap(world, now))
            intents.extend(self._handle_combat_wait(world, now, elapsed))
            return intents

        intents: list[ActionIntent] = (
            [] if self.state == BotState.DEAD_RESPAWN else self._check_survival_potions(world, now)
        )
        if self.state in (BotState.SEARCH_MOB, BotState.AT_SPOT_IDLE):
            intents.extend(self._check_weapon_swap(world, now))

        elapsed = max(0.0, now - self._state_entered_at)

        if self.state == BotState.DEAD_RESPAWN:
            rt_intents, nxt, rsn = self.respawn_travel.step_dead_respawn(world, now, elapsed)
            intents.extend(rt_intents)
            if nxt is not None:
                if (
                    nxt == BotState.TELEPORT_TO_SPOT
                    and self.respawn_travel.repair_enabled
                    and self._active_farming_sec >= (self.respawn_travel.repair_interval_sec * 0.85)
                ):
                    target_city = BotState.REPAIR_AT_NPC
                    target_rsn = "respawned_in_town_opportunistic_repair"
                else:
                    target_city = nxt
                    target_rsn = rsn
                if self._ensure_mode_enabled and target_city in (
                    BotState.REPAIR_AT_NPC,
                    BotState.TELEPORT_TO_SPOT,
                    BotState.TRAVEL_TO_SPOT,
                ):
                    self._enter_ensure_keyboard_mode(target_city, now, f"{target_rsn}_ensure_kb")
                else:
                    self.transition(target_city, now, target_rsn)
        elif self.state == BotState.REPAIR_RECALL:
            rt_intents, nxt, rsn = self.respawn_travel.step_repair_recall(world, now, elapsed)
            intents.extend(rt_intents)
            if nxt is not None:
                if self._ensure_mode_enabled and nxt == BotState.REPAIR_AT_NPC:
                    self._enter_ensure_keyboard_mode(nxt, now, f"{rsn}_ensure_kb")
                else:
                    self.transition(nxt, now, rsn)
        elif self.state == BotState.REPAIR_AT_NPC:
            rt_intents, nxt, rsn = self.respawn_travel.step_repair_at_npc(world, now)
            intents.extend(rt_intents)
            if nxt is not None:
                if nxt == BotState.TELEPORT_TO_SPOT:
                    self.repairs_count += 1
                    self._active_farming_sec = 0.0
                    self._repair_weapon_checked = False
                self.transition(nxt, now, rsn)
        elif self.state == BotState.TELEPORT_TO_SPOT:
            rt_intents, nxt, rsn = self.respawn_travel.step_teleport_to_spot(world, now)
            intents.extend(rt_intents)
            if nxt is not None:
                self.transition(nxt, now, rsn)
        elif self.state == BotState.TRAVEL_TO_SPOT:
            spot_c = self.configured_spot_center
            rt_intents, nxt, rsn = self.respawn_travel.step_travel_to_spot(
                world,
                now,
                travel_waypoints=self.navigator.route.travel_waypoints,
                spot_center=spot_c,
                farm_radius=self.navigator.route.farm_radius,
            )
            intents.extend(rt_intents)
            if nxt is not None:
                if nxt in (BotState.SEARCH_MOB, BotState.AT_SPOT_IDLE):
                    self.navigator.set_dynamic_anchor(spot_c)
                    self._need_dynamic_anchor = False
                    if self._ensure_mode_enabled:
                        self._enter_ensure_mouse_mode(nxt, now, "arrived_spot_ensure_mouse_mode")
                    else:
                        self.transition(nxt, now, rsn)
                else:
                    self.transition(nxt, now, rsn)
        elif self.state == BotState.NAVIGATE_TO_SPOT:
            intents.extend(self._handle_navigate(world, now))
        elif self.state in (BotState.SEARCH_MOB, BotState.AT_SPOT_IDLE):
            intents.extend(self._handle_search(world, now))
        elif self.state == BotState.COMBAT_WAIT:
            intents.extend(self._handle_combat_wait(world, now, elapsed))
        elif self.state == BotState.LOOT:
            intents.extend(self._handle_loot(world, now, elapsed))
        elif self.state == BotState.RETURN_TO_CENTER:
            intents.extend(self._handle_return_to_center(world, now, elapsed))
        elif self.state == BotState.RECOVER_STUCK:
            intents.extend(self._handle_recover(world, now, elapsed))

        return intents

    def _handle_ensure_mouse_mode(self, world: WorldState, now: float) -> list[ActionIntent]:
        """One-shot startup Mouse Mode verification: ESC -> 0.32s -> check banner -> optional 2nd ESC -> 0.28s.

        If resuming in Green Wales town or near Ardun Keep, immediately redirects to ENSURE_KEYBOARD_MODE instead!
        """
        esc_key = str(self.config.respawn_travel.get("mode_toggle_key", "esc"))
        if self._ensure_mouse_phase == "SEND_FIRST_ESC":
            # Smart Resume intercept: NEVER run ENSURE_MOUSE_MODE if starting in town or Ardun Keep!
            if (
                self._just_resumed
                and world.coords_smoothed is not None
                and self.mode == BotMode.FULL
                and self.respawn_travel.enabled
            ):
                if self.respawn_travel.is_in_city(world.coords_smoothed):
                    self._just_resumed = False
                    self._need_dynamic_anchor = False
                    if self._force_repair_requested or (
                        self.respawn_travel.repair_enabled
                        and self._active_farming_sec >= self.respawn_travel.repair_interval_sec
                    ):
                        self._force_repair_requested = False
                        self._enter_ensure_keyboard_mode(BotState.REPAIR_AT_NPC, now, "resumed_in_town_ensure_kb_repair")
                    else:
                        self._enter_ensure_keyboard_mode(BotState.TELEPORT_TO_SPOT, now, "resumed_in_town_ensure_kb_tele")
                    return self._handle_ensure_keyboard_mode(world, now)
                if self.respawn_travel.is_near_addon_keep(world.coords_smoothed):
                    self._just_resumed = False
                    self._need_dynamic_anchor = False
                    self.tracker.travel_waypoint_index = self.respawn_travel.choose_starting_travel_index(
                        world.coords_smoothed, self.navigator.route.travel_waypoints
                    )
                    self._enter_ensure_keyboard_mode(BotState.TRAVEL_TO_SPOT, now, "resumed_at_addon_keep_ensure_kb")
                    return self._handle_ensure_keyboard_mode(world, now)

            self._ensure_mouse_phase = "WAIT_BANNER"
            self._ensure_mouse_until = now + 0.32
            logger.info("ENSURE_MOUSE_MODE: sending initial ESC probe and waiting 0.32s for INTERFACE TYPE banner")
            return [TapKeyIntent(key=esc_key, hold_sec=0.05, reason="ensure_mouse_mode_esc_1")]

        if self._ensure_mouse_phase == "WAIT_BANNER":
            if now < self._ensure_mouse_until:
                return []
            imode = world.snapshot.interface_mode
            if imode == "keyboard_banner":
                self._ensure_mouse_phase = "WAIT_SECOND_ESC"
                self._ensure_mouse_until = now + 0.28
                logger.info("ENSURE_MOUSE_MODE: banner shows Keyboard -> pressing ESC 2nd time and waiting 0.28s")
                return [TapKeyIntent(key=esc_key, hold_sec=0.05, reason="ensure_mouse_mode_esc_2")]
            self._mouse_mode_verified = True
            self.transition(self._post_ensure_state, now, f"mouse_mode_verified({imode})")
            return []

        if now < self._ensure_mouse_until:
            return []
        self._mouse_mode_verified = True
        self.transition(self._post_ensure_state, now, "mouse_mode_verified_after_esc_2")
        return []

    def _handle_ensure_keyboard_mode(self, world: WorldState, now: float) -> list[ActionIntent]:
        """One-shot Keyboard Mode verification (inverse of ENSURE_MOUSE_MODE):
        ESC -> 0.32s -> if mouse_banner then 2nd ESC -> 0.28s -> target town/repair/teleport/travel state.
        """
        esc_key = str(self.config.respawn_travel.get("mode_toggle_key", "esc"))
        if self._ensure_kb_phase == "SEND_FIRST_ESC":
            self._ensure_kb_phase = "WAIT_BANNER"
            self._ensure_kb_until = now + 0.32
            logger.info("ENSURE_KEYBOARD_MODE: sending initial ESC probe and waiting 0.32s for INTERFACE TYPE banner")
            return [TapKeyIntent(key=esc_key, hold_sec=0.05, reason="ensure_keyboard_mode_esc_1")]

        if self._ensure_kb_phase == "WAIT_BANNER":
            if now < self._ensure_kb_until:
                return []
            imode = world.snapshot.interface_mode
            if imode == "mouse_banner":
                self._ensure_kb_phase = "WAIT_SECOND_ESC"
                self._ensure_kb_until = now + 0.28
                logger.info("ENSURE_KEYBOARD_MODE: banner shows Mouse -> pressing ESC 2nd time and waiting 0.28s")
                return [TapKeyIntent(key=esc_key, hold_sec=0.05, reason="ensure_keyboard_mode_esc_2")]
            self.transition(self._post_ensure_kb_state, now, f"keyboard_mode_verified({imode})")
            return []

        if now < self._ensure_kb_until:
            return []
        self.transition(self._post_ensure_kb_state, now, "keyboard_mode_verified_after_esc_2")
        return []

    def _handle_navigate(self, world: WorldState, now: float) -> list[ActionIntent]:
        # Priority 0: If within 85 coord units of spot center and a safe Gold Rustler is visible,
        # engage it directly instead of rotating camera / patrolling!
        if (
            self.mode == BotMode.FULL
            and world.coords_smoothed is not None
            and world.coords_smoothed.distance_to(self.navigator.spot_center) <= 85.0
            and world.mobs
        ):
            fw = max(1, world.snapshot.frame_width)
            scale_x = fw / 1920.0
            mob = self.farm.pick_best_gold_rustler(
                world,
                allow_wounded=False,
                is_ignored=lambda m: (
                    self._is_aborted_mob(m, now, scale_x)
                    or self._is_in_anti_ks_zone(m.click_x, m.click_y, world, now)
                ),
            )
            if mob is not None:
                self.tracker.clear_stuck(reset_consecutive=True)
                return self._enter_combat_wait(world, mob, now, "nav_spotted_gold_rustler_near_spot")

        if world.is_stuck:
            self.transition(BotState.RECOVER_STUCK, now, "stuck_navigating")
            return self.release_move_keys("nav_stuck_release")

        rep_intents = self._check_periodic_repair_due(world, now)
        if rep_intents is not None:
            return rep_intents

        if self.navigator.is_arrived(world.coords_smoothed):
            if self.mode == BotMode.NAVIGATE:
                self.navigator.advance_waypoint()
                return self.navigator.step_to_waypoint_keyboard(world, now)
            self.transition(BotState.SEARCH_MOB, now, "arrived_at_spot")
            return self.release_move_keys("arrived_at_spot")

        # Stop NAVIGATE_TO_SPOT micro-jitter when already within <= 12 units of spot center (594, 1832) or dynamic anchor!
        if self.mode != BotMode.NAVIGATE and world.coords_smoothed is not None:
            center_pt = Point2D(self.spot_center_x, self.spot_center_y)
            if (
                world.coords_smoothed.distance_to(center_pt) <= self.spot_center_arrival_radius
                or world.coords_smoothed.distance_to(self.navigator.spot_center) <= self.spot_center_arrival_radius
            ):
                self.transition(BotState.SEARCH_MOB, now, "arrived_within_spot_center_12u")
                return self.release_move_keys("arrived_within_spot_center_12u")

        return self.navigator.step_to_waypoint_keyboard(world, now)

    def _handle_search(self, world: WorldState, now: float) -> list[ActionIntent]:
        fw = max(1, world.snapshot.frame_width)
        scale_x = fw / 1920.0
        # Record external floating damage numbers observed while scanning (Anti-KS stealth Sin guard!)
        self._update_external_combat_zones(world, now)

        # Priority 1: BotState.LOOT is strictly entered ONLY right after COMBAT_WAIT kills a monster
        # (except standalone --mode loot CLI). Never enter LOOT from SEARCH_MOB in FULL mode!
        if self.mode == BotMode.LOOT and (world.pending_loot or self.farm.select_local_loot(world)):
            self.transition(BotState.LOOT, now, "gold_bags_waiting")
            return []

        # Priority 2: Scheduled 80m Auto-Repair once loot is clean
        rep_intents = self._check_periodic_repair_due(world, now)
        if rep_intents is not None:
            return rep_intents

        # Priority 3: Stay inside farming spot leash (block drifted_outside_spot_area when mobs>=1 unless drift > 85u!)
        spot_dist = (
            world.coords_smoothed.distance_to(self.navigator.spot_center)
            if world.coords_smoothed is not None
            else 9999.0
        )
        if (
            self.mode == BotMode.FULL
            and self.navigator.is_outside_leash_exit(world.coords_smoothed)
            and not (len(world.mobs) >= 1 and spot_dist <= 85.0)
        ):
            self.tracker.waypoint_index = 0
            self.transition(BotState.NAVIGATE_TO_SPOT, now, "drifted_outside_spot_area")
            return self.release_move_keys("spot_to_nav_release")

        # Priority 3b: Safe daylight weather lock (/setweather 11) ONLY while idle/scanning in SEARCH_MOB
        weather_intents = self.farm.maybe_build_daylight_weather_cmd(world, now)
        if weather_intents:
            return weather_intents

        # Wait settle window after MMB camera drag so we never click stale pre-drag coordinates
        if now < self._post_drag_settle_until:
            return []

        # Priority 3c: Safe vertical camera pitch auto-level (dx=0, 4s cooldown, 12s opposite lockout)
        pitch_intents, pitch_settling = self.farm.maybe_build_camera_pitch_adjust(world, now)
        if pitch_intents:
            self._post_drag_settle_until = now + self.farm.pitch_settle_sec
            return pitch_intents
        if pitch_settling:
            return []

        # Priority 4: Pick nearest safe Gold Rustler (outside Anti-KS player/damage zones & HP >= 94%) and CLICK ONCE!
        mob = self.farm.pick_best_gold_rustler(
            world,
            allow_wounded=(self.mode == BotMode.COMBAT),
            is_ignored=lambda m: (
                self._is_aborted_mob(m, now, scale_x)
                or self._is_in_anti_ks_zone(m.click_x, m.click_y, world, now)
            ),
        )
        if mob is not None:
            return self._enter_combat_wait(world, mob, now, "found_gold_rustler")

        # Priority 5: Screen empty -> rotate camera horizontally with Middle Mouse drag
        # If still no gold_rustler after 1x camera rotation and away from center -> RETURN_TO_CENTER!
        sweep = self.farm.build_mmb_camera_sweep(world, now)
        if sweep:
            if any(isinstance(a, DragCameraIntent) for a in sweep):
                self._post_drag_settle_until = now + 0.18
            self._sweeps_in_search += 1
            if self.mode == BotMode.FULL and not world.mobs:
                center_pt = Point2D(self.spot_center_x, self.spot_center_y)
                dist_c = (
                    min(
                        world.coords_smoothed.distance_to(center_pt),
                        world.coords_smoothed.distance_to(self.navigator.spot_center),
                    )
                    if world.coords_smoothed is not None
                    else 0.0
                )
                if (
                    self._sweeps_in_search >= 2
                    and world.hp_ratio > 0.01
                    and dist_c > self.spot_center_arrival_radius
                ):
                    self.transition(BotState.RETURN_TO_CENTER, now, "no_mob_after_1x_camera_sweep")
                    return []
                if self._sweeps_in_search >= self.max_sweeps_before_patrol:
                    self.navigator.advance_waypoint()
                    self.transition(BotState.NAVIGATE_TO_SPOT, now, "patrol_next_spot_point")
                    return []
        return sweep

    def _handle_combat_wait(self, world: WorldState, now: float, elapsed: float) -> list[ActionIntent]:
        """Wait calmly after 1x click with ZERO extra clicks until target dies, KS occurs, or watchdog triggers."""
        scale_x = max(0.55, world.snapshot.frame_width / 1920.0)

        # 1. Anti-KS: locked mob flagged contested by another player -> abort immediately!
        if world.locked_mob_contested_ks:
            logger.info("Anti-KS: locked target contested -> releasing lock")
            self._record_aborted_mob(world.locked_mob, now, ttl_sec=6.0)
            self.tracker.clear_target_lock()
            self.transition(BotState.SEARCH_MOB, now, "anti_ks_contested_drop")
            return []

        # 2. Update movement & active approach distance progress (chase_in_progress!)
        if world.coords_smoothed is not None:
            if self._combat_anchor_pos is None:
                self._combat_anchor_pos = world.coords_smoothed
                self._combat_last_move_ts = now
            elif world.coords_smoothed.distance_to(self._combat_anchor_pos) >= 0.8:
                self._combat_anchor_pos = world.coords_smoothed
                self._combat_last_move_ts = now
                # While character is actively running toward target, NEVER accumulate stagnant HP time!
                self._combat_last_hp_drop_ts = now

        hp_vis = False
        hp_val: float | None = None
        if world.locked_mob is not None and world.locked_mob_visible:
            lm = world.locked_mob
            self._combat_click_xy = (int(lm.click_x), int(lm.click_y))
            # Track shrinking screen distance to player so Watchdog Anti-Miss NEVER cancels an active run-up!
            if float(lm.dist_to_player_px) <= (self._combat_min_dist_px - 4.0 * scale_x):
                self._combat_min_dist_px = float(lm.dist_to_player_px)
                self._combat_last_approach_ts = now
            if lm.has_hp_bar:
                hp_vis = True
                hp_val = lm.hp_ratio_est
        elif world.target_hp_visible:
            # When all object detectors are intentionally off (COMBAT_WAIT), rely strictly on target HP probe
            hp_val = world.snapshot.combat_target_hp_ratio
            hp_vis = hp_val is None or hp_val > 0.0

        if hp_vis:
            # Anti-KS First-HP Verify: fresh target already < 72% HP on first reading -> another player hit it!
            if hp_val is not None and not self._combat_first_hp_checked:
                self._combat_first_hp_checked = True
                if (
                    self.mode == BotMode.FULL
                    and self._combat_started_fresh
                    and 0.25 <= elapsed <= 1.20
                    and hp_val < 0.72
                    and not world.loot
                ):
                    logger.info("Anti-KS: first HP reading already %.0f%% -> aborting", hp_val * 100.0)
                    self._record_aborted_mob(world.locked_mob, now, ttl_sec=6.0)
                    self.tracker.clear_target_lock()
                    self.transition(BotState.SEARCH_MOB, now, "anti_ks_damaged_on_verify")
                    return []

            if not self._combat_saw_hp:
                self._combat_last_hp_drop_ts = now
            self._combat_saw_hp = True
            self._combat_last_hp_ts = now
            if hp_val is not None and hp_val <= (self._combat_min_hp - 0.01):
                self._combat_min_hp = hp_val
                self._combat_last_hp_drop_ts = now
                if not self._combat_confirmed:
                    self._combat_confirmed = True
                    self._combat_confirmed_at = now

        # 2b. Local floating hit damage confirmation near character (x=340..685, y=190..465):
        #     Once detected -> combat_confirmed = True & refresh last_local_hit_time!
        if world.local_hit_damage_visible:
            if not self._combat_confirmed:
                self._combat_confirmed = True
                self._combat_confirmed_at = now
                logger.info("COMBAT_WAIT: local hit damage confirmed at elapsed=%.2fs -> locking combat", elapsed)
            self._combat_last_hit_time = now

        # 2c. Stage 2 (Hit-Confirmed Combat): once last local hit stops for >= 5.0s (or 35s max cap),
        #     MANDATORY transition to BotState.LOOT (10.0s) before returning to SEARCH_MOB!
        if self._combat_confirmed and self._combat_last_hit_time is not None:
            post_hit_idle = max(0.0, now - self._combat_last_hit_time)
            confirmed_dur = max(0.0, now - (self._combat_confirmed_at or self._state_entered_at))
            if post_hit_idle >= self.combat_post_last_hit_idle_sec or confirmed_dur >= self.combat_confirmed_max_sec:
                self.kills_count += 1
                self.tracker.notify_kill()
                self.tracker.record_kill_pos(
                    self._combat_click_xy,
                    now,
                    window_sec=self.last_kill_loot_window_sec,
                    radius_px=self.last_kill_loot_radius_px,
                )
                self.tracker.clear_target_lock()
                if self.mode == BotMode.COMBAT:
                    self.tracker.mark_loot_done()
                    self.transition(BotState.SEARCH_MOB, now, "hit_confirmed_kill_combat_only")
                else:
                    self.tracker.trigger_pending_loot()
                    rsn = (
                        "hit_confirmed_post_hit_idle_mandatory_loot"
                        if post_hit_idle >= self.combat_post_last_hit_idle_sec
                        else "hit_confirmed_max_35s_mandatory_loot"
                    )
                    self.transition(BotState.LOOT, now, rsn)
                return []
            # Still actively exchanging hits -> stay silent in COMBAT_WAIT!
            return []

        # 3. Kill Trigger A: Gold Bag drop appeared near feet/kill spot ONLY AFTER target HP was seen & is now gone!
        near_bags = [b for b in world.loot if b.dist_to_player_px <= 260.0 * scale_x]
        loot_kill_ok = (
            self._combat_saw_hp
            and (not hp_vis)
            and bool(near_bags)
            and elapsed >= 0.70
        )
        self._combat_loot_streak = (self._combat_loot_streak + 1) if loot_kill_ok else 0
        confirm_need = 1 if (len(near_bags) >= 2 and not hp_vis and elapsed >= 0.90) else 2
        if self._combat_loot_streak >= confirm_need:
            self.kills_count += 1
            self.tracker.notify_kill()
            self.tracker.record_kill_pos(
                self._combat_click_xy,
                now,
                window_sec=self.last_kill_loot_window_sec,
                radius_px=self.last_kill_loot_radius_px,
            )
            next_st = BotState.SEARCH_MOB if self.mode == BotMode.COMBAT else BotState.LOOT
            self.transition(next_st, now, "gold_kill_loot_detected")
            return []

        idle_no_move = max(0.0, now - self._combat_last_move_ts)
        hp_gone_sec = (now - self._combat_last_hp_ts) if self._combat_last_hp_ts is not None else elapsed

        # 4. Kill Trigger B: Target HP bar disappeared after combat -> transition to LOOT (or SEARCH_MOB in COMBAT-only)
        if self._combat_saw_hp and (not hp_vis) and hp_gone_sec >= self.combat_zero_drop_kill_sec and idle_no_move >= 1.2:
            self.kills_count += 1
            self.tracker.record_kill_pos(
                self._combat_click_xy,
                now,
                window_sec=self.last_kill_loot_window_sec,
                radius_px=self.last_kill_loot_radius_px,
            )
            self.tracker.clear_target_lock()
            if self.mode == BotMode.COMBAT:
                self.tracker.mark_loot_done()
                self.transition(BotState.SEARCH_MOB, now, "gold_kill_zero_drops")
            else:
                self.tracker.trigger_pending_loot()
                self.transition(BotState.LOOT, now, "gold_kill_hp_gone_check_loot")
            return []

        # 5. Stage 1 (No Hit Confirmation within 4.5s) & Anti-Ghost-Lock (min 2.0s lock hold):
        #    Strictly abort COMBAT_WAIT after 4.5s if NEITHER THP nor local hit damage appeared (never freeze 35s!).
        miss_limit = max(
            self.combat_min_lock_hold_sec,
            self.combat_miss_timeout_sec if world.locked_mob_visible else self.combat_ghost_lock_sec,
        )
        approach_active = world.locked_mob_visible and ((now - self._combat_last_approach_ts) < miss_limit)
        coords_moving = idle_no_move < miss_limit
        no_hit_hard_abort = (
            (not self._combat_confirmed)
            and (not self._combat_saw_hp)
            and elapsed >= max(self.combat_min_lock_hold_sec, self.combat_no_hit_abort_sec)
            and (not approach_active)
        )
        hit_confirm_expired = (
            (not self._combat_confirmed)
            and elapsed >= max(self.combat_min_lock_hold_sec, self.combat_hit_confirm_timeout_sec)
            and (not approach_active)
        )
        ghost_idle_expired = (
            (not self._combat_saw_hp)
            and (not self._combat_confirmed)
            and elapsed >= miss_limit
            and (not approach_active)
            and (not coords_moving)
        )
        if no_hit_hard_abort or hit_confirm_expired or ghost_idle_expired:
            logger.info(
                "COMBAT_WAIT miss/ghost (elapsed=%.1fs, confirmed=%s) -> blacklisting %s for %.0fs",
                elapsed,
                self._combat_confirmed,
                self._combat_click_xy,
                self.combat_ghost_blacklist_sec,
            )
            self._record_aborted_mob(
                world.locked_mob,
                now,
                ttl_sec=self.combat_ghost_blacklist_sec,
                click_xy=self._combat_click_xy,
            )
            self.tracker.clear_target_lock()
            self.transition(
                BotState.SEARCH_MOB,
                now,
                "combat_miss_no_hit_confirm_3s" if hit_confirm_expired else "anti_ghost_lock_idle",
            )
            return []

        # 6. Watchdog Stagnant HP: DISABLED completely when combat_confirmed==True OR while character is chasing!
        chase_in_progress = idle_no_move < 2.0
        if (
            (not self._combat_confirmed)
            and (not chase_in_progress)
            and hp_vis
            and (now - self._combat_last_hp_drop_ts) >= self.combat_stagnant_hp_sec
            and idle_no_move >= 4.0
        ):
            logger.info("Watchdog Stagnant HP: no damage for %.1fs -> aborting target", now - self._combat_last_hp_drop_ts)
            self._record_aborted_mob(world.locked_mob, now, ttl_sec=6.0)
            self.tracker.clear_target_lock()
            self.transition(BotState.SEARCH_MOB, now, "watchdog_hp_stagnant")
            return []

        # 7. Hard safety timeout cap (if combat was confirmed -> mandatory LOOT; else back to SEARCH_MOB)
        cap_sec = self.combat_confirmed_max_sec if self._combat_confirmed else self.combat_hard_timeout_sec
        if elapsed >= cap_sec:
            self.tracker.clear_target_lock()
            if self._combat_confirmed and self.mode != BotMode.COMBAT:
                self.kills_count += 1
                self.tracker.trigger_pending_loot()
                self.transition(BotState.LOOT, now, "combat_confirmed_cap_mandatory_loot")
            else:
                self._record_aborted_mob(world.locked_mob, now, ttl_sec=6.0)
                self.transition(BotState.SEARCH_MOB, now, "watchdog_hard_timeout")
            return []

        # Stay completely silent — ZERO extra clicks until monster dies!
        return []

    def _should_return_to_center_after_loot(self, world: WorldState) -> bool:
        """Only walk back to (594, 1832) after LOOT if character drifted > 35 units from center."""
        if world.coords_smoothed is None:
            return False
        center_pt = Point2D(self.spot_center_x, self.spot_center_y)
        return world.coords_smoothed.distance_to(center_pt) > self.return_center_min_drift_units

    def _handle_loot(self, world: WorldState, now: float, elapsed: float) -> list[ActionIntent]:
        """Post-kill LOOT (mandatory >=2.5s): sweep dead monster point + click any gold_bag <=310px while AHK E spams.

        Never requires GOLD > 0 to vacuum drops; exits directly to SEARCH_MOB if within <=35u of (594, 1832),
        or RETURN_TO_CENTER only if drifted >35u from center.
        """
        if not self.farm.loot_mouse_click_enabled:
            if elapsed >= self.post_kill_loot_window_sec:
                self.tracker.mark_loot_done()
                if self.mode == BotMode.LOOT:
                    self._state_entered_at = now
                    return []
                next_st = (
                    BotState.RETURN_TO_CENTER
                    if self._should_return_to_center_after_loot(world)
                    else BotState.SEARCH_MOB
                )
                self.transition(next_st, now, "post_kill_e_spam_loot_done")
            return []

        # Prune expired visited_loot_grids entries (TTL 15.0s)
        self.visited_loot_grids = {
            cell: (att, exp) for cell, (att, exp) in self.visited_loot_grids.items() if now < exp
        }

        # Anti-stuck: if character has been moving toward the same loot grid for >= 1.4s, mark it visited
        # so we immediately advance to the next gold_bag pile!
        if (
            self._loot_active_grid is not None
            and (now - self._loot_active_grid_since) >= self.loot_target_move_timeout_sec
        ):
            self.visited_loot_grids[self._loot_active_grid] = (
                self.loot_max_attempts_per_bag,
                now + self.loot_visited_grid_ttl_sec,
            )
            self._loot_grid_attempts[self._loot_active_grid] = self.loot_max_attempts_per_bag
            self._loot_active_grid = None

        local_bags = self.farm.select_local_loot(world)
        grid = float(self.loot_attempt_grid_px)

        def _cell_of(b_x: float, b_y: float) -> tuple[int, int]:
            return (int(b_x // grid), int(b_y // grid))

        unvisited: list = []
        retryable: list = []
        for bag in local_bags:
            if self._is_in_anti_ks_zone(
                bag.click_x,
                bag.click_y,
                world,
                now,
                player_radius_px=self.anti_ks_player_radius_px,
                damage_radius_px=self.anti_ks_player_radius_px,
            ):
                continue
            cell = _cell_of(float(bag.click_x), float(bag.click_y))
            att = self.visited_loot_grids.get(cell, (0, 0.0))[0]
            if att == 0:
                unvisited.append(bag)
            elif att < self.loot_max_attempts_per_bag:
                retryable.append(bag)

        ordered_bags = unvisited + retryable
        bag_count = len(ordered_bags)
        self._loot_prev_count = bag_count
        if bag_count > 0:
            self._loot_saw_any_bag = True

        if bag_count == 0 and not world.snapshot.pickup_log_changed:
            self._loot_clear_streak += 1
            if self._loot_empty_since is None:
                self._loot_empty_since = now
        else:
            self._loot_clear_streak = 0
            self._loot_empty_since = None

        # Priority 2 fallback sweep active when GOLD == 0 on entry (stages 0..3: corpse -> -85 -> +85 -> rescan)
        fallback_active = (
            self.loot_dead_point_sweep
            and (not self._loot_saw_any_bag)
            and bag_count == 0
            and self._loot_fallback_stage < 4
        )

        empty_dur = (now - self._loot_empty_since) if self._loot_empty_since is not None else 0.0
        window_expired = elapsed >= self.post_kill_loot_window_sec
        empty_cleared = (
            (not fallback_active)
            and elapsed >= self.loot_min_window_sec
            and empty_dur >= self.loot_empty_exit_sec
        )

        if window_expired or empty_cleared:
            self.tracker.mark_loot_done()
            if self.mode == BotMode.LOOT:
                self._state_entered_at = now
                self._loot_clear_streak = 0
                self._loot_empty_since = None
                return []
            need_rtc = self._should_return_to_center_after_loot(world)
            next_st = BotState.RETURN_TO_CENTER if need_rtc else BotState.SEARCH_MOB
            self.transition(
                next_st,
                now,
                (
                    "post_kill_loot_cleared_return_center"
                    if (empty_cleared and need_rtc)
                    else ("post_kill_loot_cleared" if empty_cleared else "post_kill_loot_window_timeout")
                ),
            )
            return []

        intents, target_bag, act_kind = self.farm.build_loot_step(world, now, candidate_bags=ordered_bags)
        if fallback_active and now >= self._loot_fallback_next_ts:
            # Priority 2 (GOLD == 0 fallback sweep): sequential 0.75s left-clicks to:
            #   Stage 0: dead_rustler / last_kill_pos / last combat click
            #   Stage 1: (-85, 0) relative to screen center (512, 384)
            #   Stage 2: (+85, 0) relative to screen center (512, 384)
            #   Stage 3: 1x final detect_loot_tiled rescan wait -> Stage 4 (done)
            fw = max(1, world.snapshot.frame_width)
            fh = max(1, world.snapshot.frame_height)
            sx_1024 = 1.0 if fw <= 1100 else (fw / 1024.0)
            max_r = float(self.config.farming.get("loot_close_radius_px", 310.0)) * sx_1024
            cx_scr = int(round(fw * 0.5))
            cy_scr = int(round(fh * 0.5))
            off_x = int(round(float(self.loot_fallback_sweep_offset_px) * sx_1024))

            click_pt: tuple[int, int, str] | None = None
            if self._loot_fallback_stage == 0:
                # 3-class YOLO fallback: use last locked gold_rustler kill coordinates (last_kill_screen_xy)
                kpos = world.last_kill_pos or self._combat_click_xy
                if kpos is not None and math.hypot(kpos[0] - cx_scr, kpos[1] - cy_scr) <= max_r:
                    click_pt = (int(kpos[0]), int(kpos[1]), "loot_fallback_kill_pos")
                self._loot_fallback_stage = 1
            elif self._loot_fallback_stage == 1:
                click_pt = (max(0, cx_scr - off_x), cy_scr, "loot_fallback_sweep_left")
                self._loot_fallback_stage = 2
            elif self._loot_fallback_stage == 2:
                click_pt = (min(fw - 1, cx_scr + off_x), cy_scr, "loot_fallback_sweep_right")
                self._loot_fallback_stage = 3
            else:
                # Stage 3 complete: final tiled rescan finished
                self._loot_fallback_stage = 4

            if click_pt is not None:
                mx, my, rsn = click_pt
                if not self._is_in_anti_ks_zone(
                    float(mx),
                    float(my),
                    world,
                    now,
                    player_radius_px=self.anti_ks_player_radius_px,
                    damage_radius_px=self.anti_ks_player_radius_px,
                ):
                    intents.append(
                        ClickScreenIntent(
                            x=mx,
                            y=my,
                            button="left",
                            hold_sec=self.loot_click_hold_sec,
                            pause_e_spam_sec=self.loot_click_pause_e_sec,
                            no_jitter=True,
                            reason=f"{rsn}({mx},{my})",
                        )
                    )
                self._loot_fallback_next_ts = now + self.loot_fallback_step_interval_sec
            elif self._loot_fallback_stage <= 3:
                self._loot_fallback_next_ts = now + self.loot_fallback_step_interval_sec

        if target_bag is not None and act_kind in ("tap_e", "click_bag"):
            bx, by = float(target_bag.click_x), float(target_bag.click_y)
            cell = _cell_of(bx, by)
            if self._loot_active_grid != cell:
                self._loot_active_grid = cell
                self._loot_active_grid_since = now
            prev_att = self.visited_loot_grids.get(cell, (0, 0.0))[0]
            attempts = prev_att + 1
            self.visited_loot_grids[cell] = (attempts, now + self.loot_visited_grid_ttl_sec)
            self._loot_grid_attempts[cell] = attempts

            if attempts >= self.loot_max_attempts_per_bag:
                logger.info(
                    "Loot grid cap (%dx) at (%.0f, %.0f) grid=%s -> blacklisted 15s/120s",
                    attempts,
                    bx,
                    by,
                    cell,
                )
                self.last_blacklisted_loot = (bx, by)
                if bag_count <= 1 and self.mode == BotMode.FULL:
                    self.tracker.mark_loot_done()
                    next_post = (
                        BotState.SEARCH_MOB
                        if (
                            self._force_repair_requested
                            or (
                                self.respawn_travel.repair_enabled
                                and self._active_farming_sec >= self.respawn_travel.repair_interval_sec
                            )
                        )
                        else BotState.RETURN_TO_CENTER
                    )
                    self.transition(next_post, now, "loot_bag_blacklisted_max_attempts")
                    return []

        return intents

    def _handle_return_to_center(self, world: WorldState, now: float, elapsed: float) -> list[ActionIntent]:
        """Walk back to spot center (594, 1832) after LOOT while AHK background E spam stays active.

        Exits to SEARCH_MOB once distance <= spot_center_arrival_radius (8u) or return_center_timeout_sec (6.0s).
        """
        rep_intents = self._check_periodic_repair_due(world, now)
        if rep_intents is not None:
            return rep_intents

        center_pt = Point2D(self.spot_center_x, self.spot_center_y)
        if (
            world.coords_smoothed is not None
            and world.coords_smoothed.distance_to(center_pt) <= self.spot_center_arrival_radius
        ):
            self.transition(BotState.SEARCH_MOB, now, "returned_to_spot_center")
            return self.release_move_keys("return_center_arrived")

        if elapsed >= self.return_center_timeout_sec:
            self.transition(BotState.SEARCH_MOB, now, "return_center_timeout")
            return self.release_move_keys("return_center_timeout_release")

        return self.navigator.step_toward_point(
            world,
            now,
            center_pt,
            label="spot_center",
            arrival_radius=self.spot_center_arrival_radius,
        )

    def _handle_recover(self, world: WorldState, now: float, elapsed: float) -> list[ActionIntent]:
        if elapsed >= self.recovery_duration_sec:
            self.tracker.clear_stuck(reset_consecutive=False)
            if not self.navigator.is_outside_farm_radius(world.coords_smoothed):
                next_state = BotState.SEARCH_MOB
            else:
                self.tracker.waypoint_index = 0
                next_state = BotState.NAVIGATE_TO_SPOT
            self.transition(next_state, now, "recovery_complete")
            return self.release_move_keys("recovery_complete_release")
        return self.navigator.build_recovery_step(world, now, self._recovery_attempts)
