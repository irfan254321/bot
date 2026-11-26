"""Unit tests for Mouse Mode navigation math, heading fusion, and stuck detection."""

from __future__ import annotations

import pytest

from vikings_bot.brain.navigation import MouseModeNavigator, world_bearing_deg
from vikings_bot.config import load_config
from vikings_bot.input.actions import ClickScreenIntent, HoldKeyIntent, MoveMouseRelIntent
from vikings_bot.state.models import CoordReading, MinimapReading, PerceptionSnapshot, Point2D
from vikings_bot.state.tracker import WorldStateTracker, wrap_angle_deg


def _make_snapshot(ts: float, x: int, y: int) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        timestamp=ts,
        frame_width=1920,
        frame_height=1080,
        hp_ratio=1.0,
        mp_ratio=1.0,
        coords=CoordReading(x=x, y=y, confidence=0.95, raw_text=f"{x},{y}"),
        minimap=MinimapReading(),
    )


def test_world_bearing_cardinal_directions():
    origin = Point2D(600.0, 1800.0)
    assert world_bearing_deg(origin, Point2D(620.0, 1800.0)) == pytest.approx(0.0)
    assert world_bearing_deg(origin, Point2D(600.0, 1820.0)) == pytest.approx(90.0)
    assert world_bearing_deg(origin, Point2D(600.0, 1780.0)) == pytest.approx(-90.0)
    assert abs(world_bearing_deg(origin, Point2D(580.0, 1800.0))) == pytest.approx(180.0)


def test_ground_click_projection_matches_four_quadrants():
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker()
    nav = MouseModeNavigator(cfg, tracker)

    cx = 1920 * 0.50
    cy = 1080 * 0.52

    # Forward (0 deg) -> above player center (-y)
    fx, fy = nav.project_ground_click(0.0, 1920, 1080)
    assert abs(fx - cx) <= 2 and fy < cy - 100

    # Right (-90 deg) -> right of player center (+x)
    rx, ry = nav.project_ground_click(-90.0, 1920, 1080)
    assert rx > cx + 150 and abs(ry - cy) <= 2

    # Left (+90 deg) -> left of player center (-x)
    lx, ly = nav.project_ground_click(90.0, 1920, 1080)
    assert lx < cx - 150 and abs(ly - cy) <= 2

    # Behind (180 deg) -> below player center (+y)
    bx, by = nav.project_ground_click(180.0, 1920, 1080)
    assert abs(bx - cx) <= 2 and by > cy + 100


def test_closed_loop_heading_calibration_and_stuck_detection():
    tracker = WorldStateTracker(stuck_timeout_sec=3.0, stuck_min_move_units=2.5)
    # Initial position at (600, 1800)
    tracker.update(_make_snapshot(10.0, 600, 1800))

    # Suppose bot clicked straight up on screen (rel=0), and character actually moved East (+X)
    # That means camera is actually facing East (0 deg), not North (90 deg)!
    tracker.notify_ground_click(10.0, rel_screen_angle_deg=0.0)
    ws = tracker.update(_make_snapshot(11.0, 610, 1800))
    assert ws.heading_source == "delta"
    assert abs(wrap_angle_deg(ws.camera_heading_deg - 0.0)) < 40.0

    # Now keep moving with zero coordinate progress past stuck_timeout_sec -> is_stuck True
    ws2 = tracker.update(_make_snapshot(14.5, 610, 1800))
    assert ws2.is_stuck is True
    assert ws2.consecutive_stuck == 1


def test_navigator_steps_toward_spot_and_detects_arrival():
    # Mouse Mode (v2 default) produces ClickScreenIntent toward spot
    cfg_m = load_config("config/default.yaml")
    assert cfg_m.controls["mode"] == "mouse"
    tracker_m = WorldStateTracker()
    nav_m = MouseModeNavigator(cfg_m, tracker_m)

    xs = [w.x for w in nav_m.route.waypoints]
    ys = [w.y for w in nav_m.route.waypoints]
    assert max(xs) - min(xs) >= 25.0
    assert max(ys) - min(ys) >= 25.0

    ws_m = tracker_m.update(_make_snapshot(1.0, 580, 1795))
    m_intents = nav_m.step_to_waypoint(ws_m, 1.0)
    assert len(m_intents) == 1 and isinstance(m_intents[0], ClickScreenIntent)
    assert not nav_m.is_arrived(ws_m.coords_smoothed)

    ws_near_m = tracker_m.update(_make_snapshot(2.0, 616, 1832))
    assert nav_m.is_arrived(ws_near_m.coords_smoothed)
    assert nav_m.step_to_waypoint(ws_near_m, 2.0) == []

    # Keyboard Mode override still supports HoldKeyIntent('w') + arrival release
    cfg = load_config("config/default.yaml")
    cfg.controls["mode"] = "keyboard"
    tracker = WorldStateTracker()
    nav = MouseModeNavigator(cfg, tracker)
    ws_far = tracker.update(_make_snapshot(1.0, 580, 1795))
    intents = nav.step_to_waypoint(ws_far, 1.0)
    assert any(isinstance(i, (HoldKeyIntent, MoveMouseRelIntent)) for i in intents)
    ws_near = tracker.update(_make_snapshot(2.0, 616, 1832))
    arr_intents = nav.step_to_waypoint(ws_near, 2.0)
    assert all(isinstance(i, HoldKeyIntent) and not i.hold for i in arr_intents)


def test_keyboard_hold_w_preserves_closed_loop_heading_and_dynamic_anchor_waypoints():
    """Continuous 30Hz HoldKeyIntent('w') notifications must NOT reset displacement anchor,
    and dynamic anchor must re-center waypoints + enforce tight +-22 spot leash."""
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker()
    nav = MouseModeNavigator(cfg, tracker)

    # 1. Dynamic anchor at (613, 1839) re-centers all waypoints & flags (646, 1802) as outside spot!
    nav.set_dynamic_anchor(Point2D(613.0, 1839.0))
    assert nav.current_target_point.x == pytest.approx(613.0)
    assert nav.current_target_point.y == pytest.approx(1839.0)
    assert nav.is_outside_farm_radius(Point2D(646.0, 1802.0)) is True
    assert nav.is_outside_farm_radius(Point2D(625.0, 1845.0)) is False

    # 2. Simulate 30Hz continuous W hold moving South-East (-45 deg) while heading started at +90 deg
    t = 10.0
    tracker.update(_make_snapshot(t, 615, 1830))
    for i in range(1, 12):
        t += 0.04
        tracker.notify_ground_click(t, rel_screen_angle_deg=0.0)
        # Small incremental movement (0.4 units/tick) -> accumulates > 2.0 units across ticks
        ws = tracker.update(_make_snapshot(t, int(round(615 + i * 0.4)), int(round(1830 - i * 0.4))))

    assert ws.heading_source == "delta"
    # True movement was South-East (-45 deg); closed-loop must have snapped near -45 deg!
    assert abs(wrap_angle_deg(ws.camera_heading_deg - (-45.0))) < 25.0
