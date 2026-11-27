"""Mouse-Mode 1-Click Farming Controller (Gold Rustler engage, Gold Bag loot, MMB camera sweep)."""

from __future__ import annotations

import math
from typing import Callable

from vikings_bot.config import BotConfig
from vikings_bot.input.actions import (
    ActionIntent,
    ClickScreenIntent,
    DragCameraIntent,
    TapKeyIntent,
    TypeChatCommandIntent,
)
from vikings_bot.input.humanizer import Humanizer
from vikings_bot.state.models import LootDrop, MobTarget, WorldState
from vikings_bot.state.tracker import WorldStateTracker


class MouseFarmController:
    """Pure Mouse-Mode farming actions: 1x click mob torso, wait until dead, tap E / click Gold Bag, MMB sweep."""

    def __init__(self, config: BotConfig, tracker: WorldStateTracker) -> None:
        self.config = config
        self.tracker = tracker
        c = config.controls
        f = config.farming
        self.player_center_norm = config.player_center_norm()

        self.attack_btn: str = str(c.get("attack_button", "left"))
        self.pickup_key: str = str(c.get("pickup_key", "e"))
        self.camera_btn: str = str(c.get("camera_rotate_button", "middle"))
        self.humanizer = Humanizer(enabled=bool(c.get("humanize", True)))

        self.mob_max_engage_dist_px: float = float(f.get("mob_max_engage_dist_px", 560.0))
        self.mob_min_y_ratio: float = float(f.get("mob_min_y_ratio", 0.14))
        self.engage_click_hold_sec: float = max(0.03, float(f.get("engage_click_hold_sec", 0.12)))
        self.engage_double_click: bool = bool(f.get("engage_double_click", True))
        self.engage_skill_key: str | None = (
            str(f.get("engage_skill_key", c.get("engage_skill_key", "1")) or "").strip() or None
        )
        self.background_e_spam: bool = bool(f.get("background_e_spam", True))
        self.loot_mouse_click_enabled: bool = bool(f.get("loot_mouse_click_enabled", True))
        self.loot_click_hold_sec: float = max(0.04, float(f.get("loot_click_hold_sec", 0.10)))
        self.loot_click_pause_e_sec: float = max(0.0, float(f.get("loot_click_pause_e_sec", 0.12)))
        self.anti_ks_min_untouched_hp_ratio: float = float(
            config.perception.get("anti_ks_min_untouched_hp_ratio", 0.94)
        )
        self.loot_e_close_dist_px: float = float(f.get("loot_e_close_dist_px", 85.0))
        self.loot_e_interval_sec: float = float(f.get("loot_e_interval_sec", 0.30))
        self.loot_click_interval_sec: float = float(f.get("loot_click_interval_sec", 0.30))
        self.loot_max_local_dist_px: float = float(f.get("loot_max_local_dist_px", 360.0))
        self.loot_close_radius_px: float = float(
            f.get("loot_close_radius_px", f.get("loot_entry_close_dist_px", 310.0))
        )
        self.loot_entry_close_dist_px: float = float(f.get("loot_entry_close_dist_px", self.loot_close_radius_px))
        self.loot_entry_kill_window_sec: float = float(f.get("loot_entry_kill_window_sec", 4.0))
        self.camera_sweep_drag_px: int = int(f.get("camera_sweep_drag_px", 185))
        self.camera_sweep_interval_sec: float = float(f.get("camera_sweep_interval_sec", 0.85))
        self.auto_daylight_weather: bool = bool(f.get("auto_daylight_weather", True))
        self.daylight_weather_cmd: str = str(f.get("daylight_weather_cmd", "/setweather 11")).strip()
        self.daylight_weather_interval_sec: float = max(30.0, float(f.get("daylight_weather_interval_sec", 600.0)))
        self.daylight_dark_luma_threshold: float = float(f.get("daylight_dark_luma_threshold", 68.0))
        self.daylight_dark_cooldown_sec: float = max(10.0, float(f.get("daylight_dark_cooldown_sec", 45.0)))

        # Auto-level vertical camera pitch (with strict anti-oscillation lockout)
        self.auto_camera_pitch: bool = bool(f.get("auto_camera_pitch", True))
        self.pitch_horizon_min_y: float = float(f.get("pitch_horizon_min_y", 105.0))
        self.pitch_horizon_max_y: float = float(f.get("pitch_horizon_max_y", 300.0))
        self.pitch_sky_ratio_min: float = float(f.get("pitch_sky_ratio_min", 0.12))
        self.pitch_sky_ratio_max: float = float(f.get("pitch_sky_ratio_max", 0.42))
        self.pitch_step_dy: int = max(15, abs(int(f.get("pitch_step_dy", 60))))
        self.pitch_adjust_cooldown_sec: float = max(1.0, float(f.get("pitch_adjust_cooldown_sec", 4.0)))
        self.pitch_opposite_lockout_sec: float = max(4.0, float(f.get("pitch_opposite_lockout_sec", 12.0)))
        self.pitch_settle_sec: float = max(0.10, float(f.get("pitch_settle_sec", 0.30)))

        self._sweep_dir: int = 1
        self._sweep_dir_chosen: bool = False
        self._weather_pending_first_entry: bool = True
        self._last_weather_sent_at: float | None = None
        self._last_dark_weather_sent_at: float | None = None
        self._last_pitch_dir: int = 0  # -1 = lifted up (dy<0), +1 = lowered down (dy>0)
        self._last_pitch_time: float = -999.0
        self._pitch_settle_until: float = 0.0

    def request_daylight_weather_sync(self) -> None:
        """Schedule a daylight weather command on the next safe SEARCH_MOB / SCAN_SPOT entry."""
        self._weather_pending_first_entry = True

    def maybe_build_daylight_weather_cmd(self, world: WorldState, now: float) -> list[ActionIntent]:
        """Send `/setweather 11` via AHK v2 CHAT_CMD when safe in SEARCH_MOB / SCAN_SPOT.

        Triggers on:
        1. First entry into SEARCH_MOB after resume (F9), DEAD_RESPAWN, or TRAVEL_TO_SPOT.
        2. Periodic interval elapsed (`daylight_weather_interval_sec`, default 600s).
        3. HUD-masked `playfield_luma` drops below `daylight_dark_luma_threshold` (default 68.0)
           with a minimum 45s cooldown to avoid spamming if rejected by server.
        """
        if not self.auto_daylight_weather or not self.daylight_weather_cmd or world.playfield_luma is None:
            return []

        reason: str | None = None
        if self._weather_pending_first_entry:
            reason = "daylight_weather_initial"
        elif self._last_weather_sent_at is None or (now - self._last_weather_sent_at) >= self.daylight_weather_interval_sec:
            reason = "daylight_weather_interval"
        elif world.playfield_luma < self.daylight_dark_luma_threshold and (
            self._last_dark_weather_sent_at is None
            or (now - self._last_dark_weather_sent_at) >= self.daylight_dark_cooldown_sec
        ):
            reason = f"daylight_weather_dark(luma={world.playfield_luma:.1f})"

        if reason is None:
            return []

        self._weather_pending_first_entry = False
        self._last_weather_sent_at = now
        self._last_dark_weather_sent_at = now
        return [TypeChatCommandIntent(command=self.daylight_weather_cmd, reason=reason)]

    def maybe_build_camera_pitch_adjust(self, world: WorldState, now: float) -> tuple[list[ActionIntent], bool]:
        """Auto-level vertical camera pitch via pure vertical DRAG_MMB (dx=0) with anti-oscillation lockout.

        Returns `(intents, is_settling)`.
        - Runs ONLY in SEARCH_MOB / SCAN_SPOT when mobs==0 or pitch is extreme.
        - Enforces >=4.0s cooldown between vertical drags, 0.30s settle wait, and 12.0s opposite-direction lockout.
        """
        if not self.auto_camera_pitch:
            return ([], False)
        if now < self._pitch_settle_until:
            return ([], True)
        hy = world.horizon_y
        sr = world.sky_ratio
        if hy is None and sr is None:
            return ([], False)
        if (now - self._last_pitch_time) < self.pitch_adjust_cooldown_sec:
            return ([], False)

        too_down = (hy is not None and hy < self.pitch_horizon_min_y) or (sr is not None and sr < self.pitch_sky_ratio_min)
        too_up = (hy is not None and hy > self.pitch_horizon_max_y) or (sr is not None and sr > self.pitch_sky_ratio_max)
        if too_down and not too_up:
            want_dir = -1  # lift camera up toward horizon (dy = -pitch_step_dy)
        elif too_up and not too_down:
            want_dir = 1  # lower camera down toward ground (dy = +pitch_step_dy)
        else:
            return ([], False)

        # Only adjust when no mobs are visible OR camera angle is extreme
        extreme = (
            (hy is not None and (hy < 60.0 or hy > 380.0))
            or (sr is not None and (sr < 0.05 or sr > 0.55))
        )
        if world.mobs and not extreme:
            return ([], False)

        # Anti-oscillation lockout: block reversing vertical direction for 12.0s!
        if (
            self._last_pitch_dir != 0
            and want_dir == -self._last_pitch_dir
            and (now - self._last_pitch_time) < self.pitch_opposite_lockout_sec
        ):
            return ([], False)

        dy = want_dir * self.pitch_step_dy
        self._last_pitch_dir = want_dir
        self._last_pitch_time = now
        self._pitch_settle_until = now + self.pitch_settle_sec
        dir_lbl = "lift_up" if want_dir < 0 else "lower_down"
        return (
            [
                DragCameraIntent(
                    dx=0,
                    dy=dy,
                    button=self.camera_btn,
                    reason=f"camera_pitch_{dir_lbl}(hy={hy},sr={sr})",
                    est_yaw_deg=0.0,
                    start_x=world.snapshot.frame_width // 2,
                    start_y=world.snapshot.frame_height // 2,
                )
            ],
            False,
        )

    def reset_sweep_direction(self) -> None:
        self._sweep_dir_chosen = False

    def is_point_blocked_by_player(
        self,
        x: float,
        y: float,
        world: WorldState,
        pad_x: float = 28.0,
        pad_y: float = 20.0,
    ) -> bool:
        """Pre-Click Anti-PK Guard: return True if (x, y) overlaps another player's bounding box."""
        boxes = world.snapshot.other_player_boxes
        pts = world.snapshot.other_players
        if not boxes and not pts:
            return False
        sx = world.snapshot.frame_width / 1920.0
        sy = world.snapshot.frame_height / 1080.0
        px_pad = pad_x * sx
        py_pad = pad_y * sy
        for ox1, oy1, ox2, oy2 in boxes:
            if (ox1 - px_pad) <= x <= (ox2 + px_pad) and (oy1 - py_pad) <= y <= (oy2 + py_pad):
                return True
        if not boxes:
            pt_r = 68.0 * sx
            for opx, opy in pts:
                if math.hypot(x - opx, y - opy) <= pt_r:
                    return True
        return False

    def pick_best_gold_rustler(
        self,
        world: WorldState,
        *,
        allow_wounded: bool = False,
        is_ignored: Callable[[MobTarget], bool] | None = None,
    ) -> MobTarget | None:
        """Choose the single nearest safe, un-contested Gold Rustler by Euclidean distance from player center."""
        fw = max(1, world.snapshot.frame_width)
        fh = max(1, world.snapshot.frame_height)
        scale_x = 1.0 if fw <= 1100 else (fw / 1920.0)
        max_dist = self.mob_max_engage_dist_px * scale_x
        min_y = self.mob_min_y_ratio * fh
        rear_y_cut = 0.72 * fh
        cx0 = fw * 0.5
        cy0 = fh * 0.5
        ordered_mobs = sorted(
            world.mobs,
            key=lambda m: math.hypot(m.click_x - cx0, m.click_y - cy0),
        )
        rear_fallback: MobTarget | None = None

        for mob in ordered_mobs:
            if mob.contested:
                continue
            if not allow_wounded and mob.has_hp_bar and (
                mob.hp_ratio_est is None or mob.hp_ratio_est < self.anti_ks_min_untouched_hp_ratio
            ):
                continue
            if is_ignored is not None and is_ignored(mob):
                continue
            if self.is_point_blocked_by_player(mob.click_x, mob.click_y, world):
                continue
            _mx, my, _mw, _mh = mob.nameplate_bbox
            if my < min_y:
                continue
            if mob.dist_to_player_px > max_dist:
                continue
            # Never click extreme right edge (x > 955 at 1024x768, Ak'Kan/Human HUD freeze guard!)
            if float(mob.click_x) > 955.0 * (float(fw) / 1024.0):
                continue
            if float(mob.click_y) > rear_y_cut:
                if rear_fallback is None:
                    rear_fallback = mob
                continue
            return mob
        return rear_fallback

    def build_engage_click(self, mob: MobTarget) -> list[ActionIntent]:
        """Engage Gold Rustler: 120ms click-hold (+ optional 2nd click) on chest/lower hitbox + attack skill key '1'."""
        intents: list[ActionIntent] = [
            ClickScreenIntent(
                x=int(mob.click_x),
                y=int(mob.click_y),
                button=self.attack_btn,
                hold_sec=self.engage_click_hold_sec,
                reason=f"mouse_engage_hold({mob.click_x},{mob.click_y})",
                no_jitter=True,
            )
        ]
        if self.engage_double_click:
            intents.append(
                ClickScreenIntent(
                    x=int(mob.click_x),
                    y=int(mob.click_y),
                    button=self.attack_btn,
                    hold_sec=0.05,
                    reason=f"mouse_engage_confirm({mob.click_x},{mob.click_y})",
                    no_jitter=True,
                )
            )
        if self.engage_skill_key:
            intents.append(
                TapKeyIntent(
                    key=self.engage_skill_key,
                    hold_sec=0.04,
                    reason=f"engage_attack_skill({self.engage_skill_key})",
                )
            )
        return intents

    def has_recent_kill(self, world: WorldState, now: float) -> bool:
        """Return True if a monster kill was recorded within the recent loot entry window (default 4.0s)."""
        return (
            world.last_kill_ts is not None
            and 0.0 <= (now - world.last_kill_ts) <= self.loot_entry_kill_window_sec
        )

    def should_enter_loot(self, world: WorldState, now: float) -> bool:
        """Gate PHASE: LOOT entry to prevent detouring to distant yellow artifacts without a recent kill.

        Allows entering LOOT ONLY if:
        - `last_kill_pos` was recorded within the last 4.0s (`has_recent_kill`), OR
        - at least one safe Gold Bag candidate is close to player feet (`<= 220 px`).
        """
        recent_kill = self.has_recent_kill(world, now)
        if world.pending_loot and recent_kill:
            return True
        local_loot = self.select_local_loot(world)
        if not local_loot:
            return False
        if recent_kill:
            return True
        return any(bag.dist_to_player_px <= self.loot_entry_close_dist_px for bag in local_loot)

    def select_local_loot(self, world: WorldState) -> list[LootDrop]:
        """Return safe Gold Bag drops within playable chase radius (full-screen <=650px for YOLO tiled gold_bag, <=310px for OpenCV)."""
        fw = max(1, world.snapshot.frame_width)
        scale_x = 1.0 if fw <= 1100 else max(0.85, fw / 1920.0)
        lim_cv = self.loot_close_radius_px * scale_x
        lim_yolo = self.loot_max_local_dist_px * scale_x
        out: list[LootDrop] = []
        for bag in world.loot:
            lim = lim_yolo if bag.class_name == "gold_bag" else lim_cv
            if bag.dist_to_player_px > lim and not bag.near_last_kill:
                continue
            if self.is_point_blocked_by_player(bag.click_x, bag.click_y, world, pad_x=22.0, pad_y=16.0):
                continue
            out.append(bag)
        return out

    def build_loot_step(
        self,
        world: WorldState,
        now: float,
        candidate_bags: list[LootDrop] | None = None,
    ) -> tuple[list[ActionIntent], LootDrop | None, str]:
        """Direct mouse-aim + 1x left-click to each close Gold Bag (<=280px) followed by E tap (0.35s).

        Returns `(intents, target_drop, action_kind)` where `action_kind` is `'tap_e' | 'click_bag' | 'wait'`.
        """
        if not self.loot_mouse_click_enabled:
            # Silent background E-spam only mode
            return ([], None, "wait")

        local_bags = candidate_bags if candidate_bags is not None else self.select_local_loot(world)
        if not local_bags:
            if self.tracker.is_ready("loot_tap_e", now):
                self.tracker.set_cooldown("loot_tap_e", now, self.loot_e_interval_sec)
                return ([TapKeyIntent(key=self.pickup_key, reason="loot_primary_e")], None, "tap_e")
            return ([], None, "wait")

        bag = local_bags[0]
        fw = max(1, world.snapshot.frame_width)
        scale_x = max(0.55, fw / 1920.0)
        close_r = self.loot_e_close_dist_px * scale_x

        if bag.dist_to_player_px <= close_r:
            if not self.tracker.is_ready("loot_tap_e", now):
                return ([], bag, "wait")
            self.tracker.set_cooldown("loot_tap_e", now, self.loot_e_interval_sec)
            return (
                [TapKeyIntent(key=self.pickup_key, reason=f"loot_e_close(d={bag.dist_to_player_px:.0f})")],
                bag,
                "tap_e",
            )

        intents: list[ActionIntent] = []
        act_kind = "wait"
        if self.tracker.is_ready("loot_click_bag", now):
            cd = self.humanizer.jitter(self.loot_click_interval_sec, pct=0.10, lo=0.22)
            self.tracker.set_cooldown("loot_click_bag", now, cd)
            intents.append(
                ClickScreenIntent(
                    x=int(bag.click_x),
                    y=int(bag.click_y),
                    button="left",
                    hold_sec=self.loot_click_hold_sec,
                    pause_e_spam_sec=self.loot_click_pause_e_sec,
                    reason=f"loot_click_bag({bag.click_x},{bag.click_y})",
                    no_jitter=True,
                )
            )
            act_kind = "click_bag"

        if self.tracker.is_ready("loot_tap_e", now):
            self.tracker.set_cooldown("loot_tap_e", now, self.loot_e_interval_sec)
            intents.append(TapKeyIntent(key=self.pickup_key, reason="loot_followup_e"))
            if act_kind == "wait":
                act_kind = "tap_e"

        return (intents, bag, act_kind)

    def build_mmb_camera_sweep(self, world: WorldState, now: float) -> list[ActionIntent]:
        """Rotate camera horizontally via Middle Mouse Button drag (strict dy=0, zero pitch drift)."""
        if not self.tracker.is_ready("camera_sweep", now):
            return []
        sweep_mag = abs(self.camera_sweep_drag_px)
        if not self._sweep_dir_chosen:
            self._sweep_dir_chosen = True
            if world.snapshot.minimap.red_blips and world.heading_source == "delta":
                nearest = world.snapshot.minimap.red_blips[0]
                rel = (nearest.world_angle_deg - world.camera_heading_deg + 180.0) % 360.0 - 180.0
                if 24.0 <= abs(rel) <= 135.0:
                    self._sweep_dir = -1 if rel > 0 else 1

        dx = self.humanizer.jitter_int(self._sweep_dir * sweep_mag, pct=0.12, min_abs=110)
        cd = self.humanizer.jitter(self.camera_sweep_interval_sec, pct=0.15, lo=0.65)
        self.tracker.set_cooldown("camera_sweep", now, cd)
        est_yaw = -dx / 2.5
        return [
            DragCameraIntent(
                dx=dx,
                dy=0,
                button=self.camera_btn,
                reason="search_mmb_camera_sweep",
                est_yaw_deg=est_yaw,
                start_x=world.snapshot.frame_width // 2,
                start_y=world.snapshot.frame_height // 2,
            )
        ]
