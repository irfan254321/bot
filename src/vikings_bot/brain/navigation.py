"""Mouse Mode waypoint navigator using verified Vikings Online world axes (+X East, +Y North)."""

from __future__ import annotations

import math

from vikings_bot.config import BotConfig, FarmingRoute, Waypoint, roi_to_pixels
from vikings_bot.input.actions import (
    ActionIntent,
    ClickScreenIntent,
    DragCameraIntent,
    HoldKeyIntent,
    MoveMouseRelIntent,
    TapKeyIntent,
)
from vikings_bot.state.models import Point2D, WorldState
from vikings_bot.state.tracker import WorldStateTracker, wrap_angle_deg


def world_bearing_deg(src: Point2D, dst: Point2D) -> float:
    """World bearing in degrees: 0 = East (+X), 90 = North (+Y), -90 = South (-Y), +-180 = West (-X)."""
    return math.degrees(math.atan2(dst.y - src.y, dst.x - src.x))


class MouseModeNavigator:
    """Walk-to-Location controller via ground clicks projected from world bearing & camera heading."""

    def __init__(self, config: BotConfig, tracker: WorldStateTracker) -> None:
        self.config = config
        self.tracker = tracker
        self.route: FarmingRoute = config.active_route
        self.playfield_roi = config.roi("playfield")
        self.player_center_norm = config.player_center_norm()

        f = config.farming
        c = config.controls
        self.control_mode = str(c.get("mode", "keyboard")).lower()
        self.forward_key = str(c.get("move_forward_key", "w"))
        self.strafe_left_key = str(c.get("strafe_left_key", "a"))
        self.strafe_right_key = str(c.get("strafe_right_key", "d"))
        self.yaw_px_per_deg = float(c.get("kb_yaw_px_per_deg", 2.2))
        self.max_turn_px = int(c.get("kb_max_turn_px", 75))
        self._dynamic_anchor: Point2D | None = None
        self._dynamic_waypoints: list[Waypoint] | None = None
        self.move_interval = float(f.get("move_click_interval_sec", 0.55))
        self.radius_x_ratio = float(f.get("move_click_radius_x_ratio", 0.145))
        self.radius_y_ratio = float(f.get("move_click_radius_y_ratio", 0.175))
        self.camera_drag_px = int(f.get("camera_sweep_drag_px", 185))
        self.camera_btn = str(c.get("camera_rotate_button", "middle"))
        self.leash_exit_margin = float(f.get("spot_leash_exit_margin", 6.0))
        self.leash_reentry_margin = float(f.get("spot_leash_reentry_margin", 6.0))

    @property
    def active_waypoints(self) -> list[Waypoint]:
        if self._dynamic_waypoints:
            return self._dynamic_waypoints
        return self.route.waypoints

    @property
    def current_waypoint(self) -> Waypoint:
        wps = self.active_waypoints
        idx = self.tracker.waypoint_index % max(1, len(wps))
        return wps[idx]

    @property
    def current_target_point(self) -> Point2D:
        wp = self.current_waypoint
        return Point2D(wp.x, wp.y)

    def advance_waypoint(self) -> Waypoint:
        wps = self.active_waypoints
        if not wps:
            return Waypoint(name="default", x=615.0, y=1830.0)
        next_idx = self.tracker.waypoint_index + 1
        if next_idx >= len(wps):
            next_idx = 0 if self.route.loop else len(wps) - 1
        self.tracker.waypoint_index = next_idx
        return wps[next_idx]

    def distance_to_waypoint(self, coords: Point2D | None) -> float | None:
        if coords is None:
            return None
        return coords.distance_to(self.current_target_point)

    def is_arrived(self, coords: Point2D | None) -> bool:
        dist = self.distance_to_waypoint(coords)
        return dist is not None and dist <= self.route.arrival_radius

    def set_dynamic_anchor(self, coords: Point2D | None) -> None:
        self._dynamic_anchor = coords
        if coords is None:
            self._dynamic_waypoints = None
            return
        # Re-center local patrol waypoints strictly inside the +-20 spot around the F9 anchor
        pad = min(12.0, max(6.0, self.route.farm_radius * 0.52))
        self._dynamic_waypoints = [
            Waypoint(name="spot_center", x=coords.x, y=coords.y),
            Waypoint(name="spot_north_east", x=coords.x + pad, y=coords.y + pad),
            Waypoint(name="spot_south_east", x=coords.x + pad, y=coords.y - pad),
            Waypoint(name="spot_south_west", x=coords.x - pad, y=coords.y - pad),
            Waypoint(name="spot_north_west", x=coords.x - pad, y=coords.y + pad),
        ]
        self.tracker.waypoint_index = 0

    @property
    def spot_center(self) -> Point2D:
        if self._dynamic_anchor is not None:
            return self._dynamic_anchor
        if self.route.waypoints:
            w0 = self.route.waypoints[0]
            return Point2D(w0.x, w0.y)
        return self.current_target_point

    def distance_from_spot_center(self, coords: Point2D | None) -> float | None:
        if coords is None:
            return None
        return coords.distance_to(self.spot_center)

    def is_outside_farm_radius(self, coords: Point2D | None) -> bool:
        if coords is None:
            return False
        return coords.distance_to(self.spot_center) > self.route.farm_radius

    def is_outside_leash_exit(self, coords: Point2D | None, extra_margin: float = 0.0) -> bool:
        """Schmitt-trigger outer leash boundary (farm_radius + exit_margin) preventing boundary yo-yo."""
        if coords is None:
            return False
        return coords.distance_to(self.spot_center) > (self.route.farm_radius + self.leash_exit_margin + extra_margin)

    def is_inside_leash_reentry(self, coords: Point2D | None) -> bool:
        """Schmitt-trigger inner re-entry boundary (farm_radius - reentry_margin) before interrupting return walk."""
        if coords is None:
            return True
        inner_r = max(self.route.arrival_radius, self.route.farm_radius - self.leash_reentry_margin)
        return coords.distance_to(self.spot_center) <= inner_r

    def project_ground_click(
        self,
        rel_angle_deg: float,
        frame_w: int,
        frame_h: int,
        scale: float = 1.0,
    ) -> tuple[int, int]:
        """Project relative angle (0=forward/up, +90=left, -90=right) onto safe ground playfield."""
        rad = math.radians(rel_angle_deg)
        cx = frame_w * self.player_center_norm[0]
        cy = frame_h * self.player_center_norm[1]
        rx = frame_w * self.radius_x_ratio * scale
        ry = frame_h * self.radius_y_ratio * scale

        dx = -math.sin(rad) * rx
        dy = -math.cos(rad) * ry

        px1, py1, px2, py2 = roi_to_pixels(self.playfield_roi, frame_w, frame_h)
        # Extra safe padding inside playfield away from HUD panels
        tx = int(round(max(px1 + 30, min(px2 - 30, cx + dx))))
        ty = int(round(max(py1 + 40, min(py2 - 30, cy + dy))))
        return (tx, ty)

    def step_to_waypoint_keyboard(self, state: WorldState, now: float) -> list[ActionIntent]:
        """Pure 100% Keyboard Mode (WASD + Keyboard Mode camera yaw) waypoint steering. Zero ground clicks."""
        if state.coords_smoothed is None:
            return []

        target = self.current_target_point
        dist = state.coords_smoothed.distance_to(target)
        if dist <= self.route.arrival_radius:
            self.tracker.notify_movement_stopped()
            return [HoldKeyIntent(key=self.forward_key, hold=False, reason="arrived_waypoint")]

        bearing = world_bearing_deg(state.coords_smoothed, target)
        rel_angle = wrap_angle_deg(bearing - state.camera_heading_deg)

        intents: list[ActionIntent] = []
        if state.heading_source != "delta":
            intents.append(
                HoldKeyIntent(
                    key=self.forward_key,
                    hold=True,
                    reason=f"calibrate_heading->{self.current_waypoint.name} d={dist:.1f}",
                    move_rel_angle_deg=0.0,
                )
            )
            return intents

        if abs(rel_angle) > 12.0 and self.tracker.is_ready("kb_nav_turn", now):
            raw_dx = int(round(-rel_angle * self.yaw_px_per_deg * 0.55))
            dx = max(-self.max_turn_px, min(self.max_turn_px, raw_dx))
            if abs(dx) >= 4:
                est_yaw = -dx / max(1.0, self.yaw_px_per_deg)
                self.tracker.set_cooldown("kb_nav_turn", now, 0.14)
                intents.append(
                    MoveMouseRelIntent(
                        dx=dx,
                        reason=f"nav_yaw->{self.current_waypoint.name} err={rel_angle:.0f}",
                        est_yaw_deg=est_yaw,
                    )
                )
        if abs(rel_angle) <= 45.0:
            intents.append(
                HoldKeyIntent(
                    key=self.forward_key,
                    hold=True,
                    reason=f"walk->{self.current_waypoint.name} d={dist:.1f}",
                    move_rel_angle_deg=0.0,
                )
            )
        else:
            intents.append(HoldKeyIntent(key=self.forward_key, hold=False, reason="turn_toward_spot_first"))
        return intents

    def step_to_waypoint(self, state: WorldState, now: float) -> list[ActionIntent]:
        if state.coords_smoothed is None:
            return []

        if self.control_mode == "keyboard":
            return self.step_to_waypoint_keyboard(state, now)

        target = self.current_target_point
        dist = state.coords_smoothed.distance_to(target)
        if dist <= self.route.arrival_radius:
            self.tracker.notify_movement_stopped()
            return []

        bearing = world_bearing_deg(state.coords_smoothed, target)
        rel_angle = wrap_angle_deg(bearing - state.camera_heading_deg)

        # Mouse Mode ground click navigation: crisp 0.65s stride spacing (avoids sluggish 0.95s stops
        # while keeping closed-loop heading calibration smooth without rapid back-and-forth clicks!)
        if not self.tracker.is_ready("move_click", now):
            return []
        step_scale = 0.65 if dist < (self.route.arrival_radius * 2.2) else 1.0
        click_x, click_y = self.project_ground_click(
            rel_angle,
            state.snapshot.frame_width,
            state.snapshot.frame_height,
            scale=step_scale,
        )
        self.tracker.set_cooldown("move_click", now, max(self.move_interval, 0.65))
        return [
            ClickScreenIntent(
                x=click_x,
                y=click_y,
                button="left",
                reason=f"walk->{self.current_waypoint.name} d={dist:.1f}",
                ground_rel_angle_deg=rel_angle,
            )
        ]

    def step_toward_point(
        self,
        state: WorldState,
        now: float,
        target: Point2D,
        *,
        label: str = "spot_center",
        arrival_radius: float = 8.0,
    ) -> list[ActionIntent]:
        """Walk directly toward an arbitrary world coordinate (e.g. RETURN_TO_CENTER (594, 1832))."""
        pos = state.coords_smoothed
        if pos is None:
            return []
        dist = pos.distance_to(target)
        if dist <= arrival_radius:
            return []
        bearing = world_bearing_deg(pos, target)
        rel_angle = (bearing - state.camera_heading_deg + 180.0) % 360.0 - 180.0
        if not self.tracker.is_ready("move_click", now):
            return []
        step_scale = 0.65 if dist < (arrival_radius * 2.2) else 1.0
        click_x, click_y = self.project_ground_click(
            rel_angle,
            state.snapshot.frame_width,
            state.snapshot.frame_height,
            scale=step_scale,
        )
        self.tracker.set_cooldown("move_click", now, max(self.move_interval, 0.55))
        return [
            ClickScreenIntent(
                x=click_x,
                y=click_y,
                button="left",
                reason=f"return->{label} d={dist:.1f}",
                ground_rel_angle_deg=rel_angle,
            )
        ]

    def build_recovery_step(self, state: WorldState, now: float, attempt: int) -> list[ActionIntent]:
        """Unstick maneuver: sidestep + camera turn adapted for Keyboard or Mouse Mode."""
        if not self.tracker.is_ready("recovery_action", now):
            return []
        drag_dx = self.camera_drag_px if (attempt % 2 == 0) else -self.camera_drag_px
        self.tracker.set_cooldown("recovery_action", now, 0.65)

        if self.control_mode == "keyboard":
            strafe_key = self.strafe_left_key if (attempt % 2 == 0) else self.strafe_right_key
            return [
                HoldKeyIntent(key=self.forward_key, hold=False, reason="unstick_stop_w"),
                TapKeyIntent(key=strafe_key, hold_sec=0.35, reason=f"unstick_strafe_{attempt}"),
                MoveMouseRelIntent(
                    dx=drag_dx,
                    reason="unstick_kb_cam_turn",
                    est_yaw_deg=-50.0 if drag_dx > 0 else 50.0,
                ),
            ]

        rel_angle = 95.0 if (attempt % 2 == 0) else -95.0
        click_x, click_y = self.project_ground_click(
            rel_angle,
            state.snapshot.frame_width,
            state.snapshot.frame_height,
            scale=0.95,
        )
        return [
            ClickScreenIntent(
                x=click_x,
                y=click_y,
                button="left",
                reason=f"unstick_sidestep_{attempt}",
                ground_rel_angle_deg=rel_angle,
            ),
            DragCameraIntent(
                dx=drag_dx,
                button=self.camera_btn,
                reason="unstick_cam_turn",
                est_yaw_deg=-45.0 if drag_dx > 0 else 45.0,
                start_x=state.snapshot.frame_width // 2,
                start_y=state.snapshot.frame_height // 2,
            ),
        ]
