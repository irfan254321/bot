"""Temporal world-state tracker: heading fusion, stuck detection, strict target lock, last_kill_pos loot priority."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

from vikings_bot.state.models import (
    MobTarget,
    PerceptionSnapshot,
    Point2D,
    WorldState,
)
from vikings_bot.state.target_lock import StrictTargetLock


def wrap_angle_deg(angle: float) -> float:
    """Wrap angle to [-180.0, 180.0)."""
    return (angle + 180.0) % 360.0 - 180.0


def blend_angles_deg(current: float, target: float, alpha: float) -> float:
    """Smoothly blend angles on a circle."""
    diff = wrap_angle_deg(target - current)
    return wrap_angle_deg(current + alpha * diff)


class WorldStateTracker:
    """Maintains multi-frame context for navigation, respawn/repair travel, 1-click target lock, and loot priority."""

    def __init__(
        self,
        coord_max_jump: float = 120.0,
        stuck_timeout_sec: float = 4.0,
        stuck_min_move_units: float = 2.5,
        death_zero_hp_frames: int = 12,
        lock_iou_threshold: float = 0.30,
        lock_dist_threshold_px: float = 85.0,
    ) -> None:
        self.coord_max_jump = coord_max_jump
        self.stuck_timeout_sec = stuck_timeout_sec
        self.stuck_min_move_units = stuck_min_move_units
        self.death_zero_hp_frames = max(5, int(death_zero_hp_frames))

        self._coord_history: deque[tuple[float, Point2D]] = deque(maxlen=64)
        self._last_valid_coord: Point2D | None = None
        self._last_valid_coord_ts: float = 0.0
        self._pending_jump_coords: list[Point2D] = []
        self._accept_next_jump: bool = False

        self._camera_heading_deg: float = 90.0
        self._heading_source: str = "default"

        self._move_intent_active: bool = False
        self._move_intent_since: float | None = None
        self._move_anchor_coord: Point2D | None = None
        self._last_click_rel_angle_deg: float | None = None
        self._last_click_coord: Point2D | None = None

        self._is_stuck: bool = False
        self._consecutive_stuck: int = 0
        self._zero_hp_streak: int = 0
        self._last_nonzero_hp_ratio: float = 1.0
        self._low_hp_seen_at: float | None = None
        self._low_hp_window_sec: float = 15.0

        self._cooldowns: dict[str, float] = {}
        self.waypoint_index: int = 0
        self.travel_waypoint_index: int = 0
        self.last_action_summary: str = ""

        self.target_lock = StrictTargetLock(
            lock_iou_threshold=lock_iou_threshold,
            lock_dist_threshold_px=lock_dist_threshold_px,
        )
        self.last_kill_pos: tuple[int, int] | None = None
        self.last_kill_ts: float | None = None
        self.last_kill_until: float = 0.0
        self.last_kill_radius_px: float = 140.0
        self._pending_loot: bool = False

    # --- Target lock & kill-loot helpers ---
    def lock_target(self, mob: MobTarget, now: float) -> int:
        return self.target_lock.lock(mob, now)

    def clear_target_lock(self) -> None:
        self.target_lock.clear()

    def record_kill_pos(
        self,
        pos: tuple[int, int] | None,
        now: float,
        window_sec: float = 0.80,
        radius_px: float = 140.0,
    ) -> None:
        """Record death position to prioritize nearby Gold Bag drops within 140px for 800ms."""
        if pos is None:
            return
        self.last_kill_pos = (int(pos[0]), int(pos[1]))
        self.last_kill_ts = float(now)
        self.last_kill_until = float(now) + max(0.05, float(window_sec))
        self.last_kill_radius_px = float(radius_px)

    def trigger_pending_loot(self) -> None:
        self._pending_loot = True

    def mark_loot_done(self) -> None:
        self._pending_loot = False
        self.last_kill_pos = None
        self.last_kill_ts = None
        self.last_kill_until = 0.0
        self._consecutive_stuck = 0

    def notify_kill(self) -> None:
        self._consecutive_stuck = 0
        self._pending_loot = True

    # --- Cooldown helpers ---
    def is_ready(self, key: str, now: float) -> bool:
        return now >= self._cooldowns.get(key, 0.0)

    def set_cooldown(self, key: str, now: float, duration_sec: float) -> None:
        self._cooldowns[key] = now + duration_sec

    # --- Motion / navigation feedback hooks ---
    def notify_ground_click(self, now: float, rel_screen_angle_deg: float) -> None:
        was_active = self._move_intent_active
        prev_angle = self._last_click_rel_angle_deg
        self._move_intent_active = True
        if self._move_intent_since is None:
            self._move_intent_since = now
            self._move_anchor_coord = self._last_valid_coord
        angle_changed = prev_angle is None or abs(wrap_angle_deg(rel_screen_angle_deg - prev_angle)) > 5.0
        if not was_active or angle_changed or self._last_click_coord is None:
            self._last_click_coord = self._last_valid_coord
        self._last_click_rel_angle_deg = rel_screen_angle_deg

    def notify_camera_rotated(self, delta_yaw_deg: float) -> None:
        self._camera_heading_deg = wrap_angle_deg(self._camera_heading_deg + delta_yaw_deg)
        if (not self._move_intent_active) or abs(delta_yaw_deg) >= 35.0:
            self._last_click_coord = self._last_valid_coord

    def notify_camera_mouse_dx(self, dx_px: int) -> None:
        _ = dx_px

    def notify_camera_drag(self, dx_px: int) -> None:
        _ = dx_px

    def notify_movement_stopped(self) -> None:
        self._move_intent_active = False
        self._move_intent_since = None
        self._move_anchor_coord = None
        self._is_stuck = False

    def clear_stuck(self, reset_consecutive: bool = False) -> None:
        self._is_stuck = False
        self._move_intent_since = None
        self._move_anchor_coord = self._last_valid_coord
        if reset_consecutive:
            self._consecutive_stuck = 0

    def reset_for_resume(self) -> None:
        """Reset stuck counters, stuck timer, and navigation history cleanly on F9 resume."""
        self.clear_stuck(reset_consecutive=True)
        self.notify_movement_stopped()
        self._coord_history.clear()
        self._last_click_coord = None
        self._last_click_rel_angle_deg = None

    def accept_teleport_jump(self) -> None:
        self._accept_next_jump = True
        self._pending_jump_coords.clear()

    def reset_for_respawn(self) -> None:
        self.clear_target_lock()
        self.notify_movement_stopped()
        self._pending_loot = False
        self.last_kill_pos = None
        self.last_kill_ts = None
        self.last_kill_until = 0.0
        self._consecutive_stuck = 0
        self._zero_hp_streak = 0
        self._last_nonzero_hp_ratio = 1.0
        self._low_hp_seen_at = None
        self._heading_source = "default"
        self._last_click_coord = None
        self._last_click_rel_angle_deg = None
        self.travel_waypoint_index = 0
        self.accept_teleport_jump()

    def update(self, snapshot: PerceptionSnapshot) -> WorldState:
        now = snapshot.timestamp

        # 1. Coordinate update with anti-spike OCR filter & automatic respawn/teleport jump acceptance
        if snapshot.coords is not None:
            candidate = snapshot.coords.point
            accept = True
            teleported = False
            if self._last_valid_coord is not None and not self._accept_next_jump:
                dt = max(0.01, now - self._last_valid_coord_ts)
                jump_dist = self._last_valid_coord.distance_to(candidate)
                allowed_jump = self.coord_max_jump * max(1.0, dt / 1.5)
                if jump_dist > allowed_jump:
                    # Large jump (> coord_max_jump, e.g. > 40u teleport/respawn): accept on 3 consistent frames
                    if (
                        self._pending_jump_coords
                        and self._pending_jump_coords[-1].distance_to(candidate) <= 18.0
                    ):
                        self._pending_jump_coords.append(candidate)
                    else:
                        self._pending_jump_coords = [candidate]
                    if len(self._pending_jump_coords) >= 3:
                        accept = True
                        teleported = True
                        self._pending_jump_coords.clear()
                    else:
                        accept = False
                elif dt < 0.5 and 8.0 < jump_dist <= 15.0:
                    # Anti-spike filter for 1-frame minimap OCR digit misreads (e.g. 1065 -> 1075 = 10u in <0.5s):
                    # reject single outlier; accept only if confirmed on 2 consecutive frames (<= 4u apart)
                    if (
                        self._pending_jump_coords
                        and self._pending_jump_coords[-1].distance_to(candidate) <= 4.0
                    ):
                        accept = True
                        self._pending_jump_coords.clear()
                    else:
                        self._pending_jump_coords = [candidate]
                        accept = False
                else:
                    self._pending_jump_coords.clear()
            elif self._accept_next_jump:
                if (
                    self._last_valid_coord is not None
                    and self._last_valid_coord.distance_to(candidate) > self.coord_max_jump
                ):
                    teleported = True
                    self._accept_next_jump = False
                self._pending_jump_coords.clear()

            if accept:
                self._coord_history.append((now, candidate))
                self._last_valid_coord = candidate
                self._last_valid_coord_ts = now
                if teleported:
                    self._heading_source = "default"
                    self._last_click_coord = None
                    self._move_anchor_coord = candidate
                    self._move_intent_since = now
                    self._is_stuck = False
                    self._consecutive_stuck = 0

        # 2. Closed-loop camera heading fusion
        if (
            self._last_click_rel_angle_deg is not None
            and self._last_click_coord is not None
            and self._last_valid_coord is not None
        ):
            dx = self._last_valid_coord.x - self._last_click_coord.x
            dy = self._last_valid_coord.y - self._last_click_coord.y
            dist = math.hypot(dx, dy)
            if dist >= 2.0:
                move_world_angle = math.degrees(math.atan2(dy, dx))
                inferred_heading = wrap_angle_deg(move_world_angle - self._last_click_rel_angle_deg)
                alpha = 0.85 if self._heading_source != "delta" else 0.65
                self._camera_heading_deg = blend_angles_deg(self._camera_heading_deg, inferred_heading, alpha)
                self._heading_source = "delta"
                self._last_click_coord = self._last_valid_coord

        if snapshot.minimap.heading_deg is not None and snapshot.minimap.heading_confidence >= 0.85:
            if self._heading_source == "default":
                self._camera_heading_deg = snapshot.minimap.heading_deg
                self._heading_source = "minimap"

        # 3. Stuck detection while moving
        if self._move_intent_active and self._last_valid_coord is not None:
            if self._move_anchor_coord is None or self._move_intent_since is None:
                self._move_anchor_coord = self._last_valid_coord
                self._move_intent_since = now
            else:
                moved = self._move_anchor_coord.distance_to(self._last_valid_coord)
                if moved >= self.stuck_min_move_units:
                    self._move_anchor_coord = self._last_valid_coord
                    self._move_intent_since = now
                    self._is_stuck = False
                    self._consecutive_stuck = 0
                elif (now - self._move_intent_since) >= self.stuck_timeout_sec:
                    if not self._is_stuck:
                        self._consecutive_stuck += 1
                    self._is_stuck = True

        # 4. Death / zero HP streak + Anti-False-Death prior-HP trajectory tracking
        if snapshot.hp_ratio <= 0.01:
            self._zero_hp_streak += 1
        else:
            self._zero_hp_streak = 0
            self._last_nonzero_hp_ratio = float(snapshot.hp_ratio)
            if snapshot.hp_ratio < 0.35:
                self._low_hp_seen_at = now
            elif snapshot.hp_ratio >= 0.70:
                # Healed back up comfortably -> older low-HP dips must not authorize a future glitchy 0% death
                self._low_hp_seen_at = None

        hp_dropped_below_35_recently = (self._last_nonzero_hp_ratio < 0.35) or (
            self._low_hp_seen_at is not None and (now - self._low_hp_seen_at) <= self._low_hp_window_sec
        )
        respawn_dialog_visible = bool(snapshot.respawn_dialog_visible)
        confirmed_dead = (
            self._zero_hp_streak >= max(5, self.death_zero_hp_frames)
            and (respawn_dialog_visible or hp_dropped_below_35_recently)
        )

        # 5. Update strict single-target lock
        matched_lock = self.target_lock.update(
            snapshot.mobs,
            now,
            snapshot.frame_width,
            contested_mobs=snapshot.contested_mobs,
        )
        lock_contested_ks = bool(self.target_lock.state.contested_ks) if self.target_lock.state is not None else False
        lock_lost = self.target_lock.is_target_dead_or_lost(now) if self.target_lock.active else False

        # 6. Prioritize Gold Bag drops within 140px of last_kill_pos during 800ms post-kill window
        scale_x = max(0.45, snapshot.frame_width / 1920.0)
        kill_pos_active = self.last_kill_pos if now <= self.last_kill_until else None
        final_snap = snapshot
        if kill_pos_active is not None and snapshot.loot:
            kx, ky = kill_pos_active
            r_lim = self.last_kill_radius_px * scale_x
            near_kill = []
            far_kill = []
            for bag in snapshot.loot:
                d_k = math.hypot(bag.click_x - kx, bag.click_y - ky)
                if d_k <= r_lim:
                    near_kill.append((d_k, replace(bag, near_last_kill=True)))
                else:
                    far_kill.append(bag)
            if near_kill:
                near_kill.sort(key=lambda pair: pair[0])
                final_snap = replace(snapshot, loot=tuple([b for _, b in near_kill] + far_kill))

        coords_age = (
            999.0 if self._last_valid_coord is None else max(0.0, now - self._last_valid_coord_ts)
        )

        return WorldState(
            snapshot=final_snap,
            coords_smoothed=self._last_valid_coord,
            camera_heading_deg=self._camera_heading_deg,
            heading_source=self._heading_source,
            is_stuck=self._is_stuck,
            consecutive_stuck=self._consecutive_stuck,
            zero_hp_streak=self._zero_hp_streak,
            coords_age_sec=coords_age,
            waypoint_index=self.waypoint_index,
            locked_mob=self.target_lock.locked_mob,
            locked_mob_id=self.target_lock.lock_id,
            locked_mob_visible=(matched_lock is not None),
            locked_mob_lost=lock_lost,
            locked_mob_contested_ks=lock_contested_ks,
            target_hp_visible=bool(
                (matched_lock is not None and matched_lock.has_hp_bar) or final_snap.target_hp_visible
            ),
            last_kill_pos=kill_pos_active,
            last_kill_ts=self.last_kill_ts,
            last_kill_until=self.last_kill_until,
            pending_loot=self._pending_loot,
            is_dead=confirmed_dead,
            last_nonzero_hp_ratio=self._last_nonzero_hp_ratio,
            hp_dropped_below_35_recently=hp_dropped_below_35_recently,
            respawn_dialog_visible=respawn_dialog_visible,
            travel_waypoint_index=self.travel_waypoint_index,
            last_action_summary=self.last_action_summary,
        )
