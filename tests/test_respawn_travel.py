"""Unit tests for Auto-Respawn -> Town -> Addon Keep Teleport -> Multi-Waypoint Return Loop."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from vikings_bot.brain.fsm import FarmingFSM
from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import load_config
from vikings_bot.input.actions import (
    ClickScreenIntent,
    HoldKeyIntent,
    TapKeyIntent,
)
from vikings_bot.perception.ui_matcher import UiTemplateMatcher
from vikings_bot.runtime.safety import SafetyMonitor
from vikings_bot.state.models import (
    BotMode,
    BotState,
    CoordReading,
    MinimapReading,
    PerceptionSnapshot,
)
from vikings_bot.state.tracker import WorldStateTracker

SCREENSHOTS_DIR = Path(r"C:\Program Files (x86)\Vikings Online\ScreenShots")


def _snap(
    ts: float,
    x: int,
    y: int,
    hp: float = 1.0,
    ea: int | None = None,
    sign_bbox: tuple[int, int, int, int] | None = None,
    menu_bbox: tuple[int, int, int, int] | None = None,
    talk_ok_bbox: tuple[int, int, int, int] | None = None,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        timestamp=ts,
        frame_width=1920,
        frame_height=1080,
        hp_ratio=hp,
        mp_ratio=1.0,
        coords=CoordReading(x=x, y=y, confidence=0.98, raw_text=f"{x},{y}"),
        minimap=MinimapReading(),
        ea_value=ea,
        addon_keep_sign_bbox=sign_bbox,
        teleport_menu_bbox=menu_bbox,
        npc_talk_dialog_bbox=talk_ok_bbox,
    )


def _feed_death(tracker: WorldStateTracker, fsm: FarmingFSM, t_start: float, x: int, y: int, frames: int = 12) -> float:
    t = t_start
    # Realistic pre-death combat drop below 35% HP before reaching 0% (required by Anti-False-Death guard)
    fsm.step(tracker.update(_snap(t, x, y, hp=0.22)))
    t += 0.04
    for _ in range(frames):
        w = tracker.update(_snap(t, x, y, hp=0.0))
        fsm.step(w)
        t += 0.04
    return t


def test_full_death_respawn_teleport_and_travel_to_gem_spot():
    cfg = load_config("config/default.yaml", route_override="gem_rustler_spot")
    cfg.controls["mode"] = "keyboard"
    cfg.respawn_travel["mouse_only_travel"] = False
    tracker = WorldStateTracker(death_zero_hp_frames=12)
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    safety = SafetyMonitor(cfg, enable_hotkeys=False)

    fsm.resume(1.0)
    fsm.step(tracker.update(_snap(1.0, 1092, 1115)))
    assert fsm.state == BotState.SEARCH_MOB

    t = _feed_death(tracker, fsm, 2.0, 1053, 1117, frames=12)
    assert fsm.state == BotState.DEAD_RESPAWN
    assert fsm.deaths_count == 1
    chk = safety.check(tracker.update(_snap(t, 1053, 1117, hp=0.0)), is_running_active=True, in_respawn_or_travel=True)
    assert chk.force_pause_reason is None

    acts_dc1 = fsm.step(tracker.update(_snap(t + 1.6, 1053, 1117, hp=0.0)))
    assert any(isinstance(a, ClickScreenIntent) and a.button == "left" and "lclick_1" in a.reason for a in acts_dc1)
    acts_dc2 = fsm.step(tracker.update(_snap(t + 1.9, 1053, 1117, hp=0.0)))
    assert any(isinstance(a, ClickScreenIntent) and a.button == "left" and "lclick_2" in a.reason for a in acts_dc2)

    w_town = tracker.update(_snap(t + 4.0, 1075, 2628, hp=1.0))
    assert w_town.coords_smoothed is not None
    assert abs(w_town.coords_smoothed.x - 1075) < 1.0
    fsm.step(w_town)
    assert fsm.state == BotState.TELEPORT_TO_SPOT

    acts_walk = fsm.step(tracker.update(_snap(t + 4.2, 1075, 2628, hp=1.0)))
    assert any(isinstance(a, HoldKeyIntent) and a.key == "w" and a.hold for a in acts_walk)

    fsm.step(tracker.update(_snap(t + 5.5, 1060, 2626, hp=1.0)))
    assert fsm.respawn_travel._tele_phase == "FIND_SIGN"

    sign_box = (895, 434, 156, 31)
    acts_cur = fsm.step(tracker.update(_snap(t + 5.7, 1060, 2626, sign_bbox=sign_box)))
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" for a in acts_cur)

    acts_npc = fsm.step(tracker.update(_snap(t + 5.9, 1060, 2626, sign_bbox=sign_box)))
    assert any(isinstance(a, ClickScreenIntent) and 960 <= a.x <= 985 and 470 <= a.y <= 500 for a in acts_npc)
    assert fsm.respawn_travel._tele_phase == "WAIT_MENU"

    menu_box = (1026, 318, 64, 43)
    acts_menu = fsm.step(tracker.update(_snap(t + 6.3, 1060, 2626, sign_bbox=sign_box, menu_bbox=menu_box)))
    assert any(
        isinstance(a, ClickScreenIntent) and a.x == 1058 and 350 <= a.y <= 352 and a.no_jitter for a in acts_menu
    )
    assert fsm.respawn_travel._tele_phase == "WAIT_LOAD"

    acts_land = fsm.step(tracker.update(_snap(t + 11.5, 1409, 1779)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT
    assert not any(isinstance(a, TapKeyIntent) and a.key in ("tab", "esc") for a in acts_land)

    ts_tr = t + 12.0
    for wp in cfg.active_route.travel_waypoints:
        fsm.step(tracker.update(_snap(ts_tr, int(wp.x), int(wp.y))))
        ts_tr += 3.0
    assert fsm.state == BotState.SEARCH_MOB
    assert fsm.navigator.spot_center.distance_to(fsm.configured_spot_center) <= 1.0


def test_die_mid_way_during_travel_repeats_infinitely_until_spot_reached():
    cfg = load_config("config/default.yaml", route_override="gold_rustler_spot")
    tracker = WorldStateTracker(death_zero_hp_frames=6)
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)

    fsm.resume(1.0, ensure_mouse_mode=False)
    fsm.step(tracker.update(_snap(1.0, 1409, 1779)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT

    t = 2.0
    for wp in cfg.active_route.travel_waypoints[:3]:
        fsm.step(tracker.update(_snap(t, int(wp.x), int(wp.y))))
        t += 2.0
    assert tracker.travel_waypoint_index >= 3

    t = _feed_death(tracker, fsm, t, 1050, 1740, frames=6)
    assert fsm.state == BotState.DEAD_RESPAWN
    assert fsm.deaths_count == 1
    assert tracker.travel_waypoint_index == 0

    fsm.step(tracker.update(_snap(t + 4.0, 1076, 2630, hp=1.0)))
    assert fsm.state == BotState.TELEPORT_TO_SPOT
    fsm.step(tracker.update(_snap(t + 10.0, 1409, 1779, hp=1.0)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT

    t = _feed_death(tracker, fsm, t + 14.0, 900, 1725, frames=6)
    assert fsm.state == BotState.DEAD_RESPAWN
    assert fsm.deaths_count == 2
    assert tracker.travel_waypoint_index == 0

    fsm.step(tracker.update(_snap(t + 4.0, 1075, 2628, hp=1.0)))
    assert fsm.state == BotState.TELEPORT_TO_SPOT
    fsm.step(tracker.update(_snap(t + 10.0, 1409, 1779, hp=1.0)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT
    ts_ok = t + 11.0
    for wp in cfg.active_route.travel_waypoints:
        fsm.step(tracker.update(_snap(ts_ok, int(wp.x), int(wp.y), hp=1.0)))
        ts_ok += 2.0
    assert fsm.state == BotState.SEARCH_MOB
    assert fsm.deaths_count == 2


def test_travel_drinks_hp_pot_while_walking_to_spot():
    cfg = load_config("config/default.yaml", route_override="gold_rustler_spot")
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)

    fsm.resume(1.0, ensure_mouse_mode=False)
    fsm.step(tracker.update(_snap(1.0, 1409, 1779)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT

    acts = fsm.step(tracker.update(_snap(2.0, 1340, 1770, hp=0.45)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT
    assert any(isinstance(a, TapKeyIntent) and a.key == "0" for a in acts)
    assert any(isinstance(a, (ClickScreenIntent, HoldKeyIntent)) for a in acts)


def test_mouse_mode_auto_switches_to_kb_for_teleporter_walk():
    cfg = load_config("config/default.yaml", route_override="gold_rustler_spot")
    cfg.controls["mode"] = "mouse"
    cfg.respawn_travel["mouse_only_travel"] = False
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)

    fsm.resume(1.0, ensure_mouse_mode=False)
    acts = fsm.step(tracker.update(_snap(1.0, 1075, 2628)))
    assert fsm.state == BotState.TELEPORT_TO_SPOT
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" for a in acts)
    assert any(isinstance(a, HoldKeyIntent) and a.key == "w" and a.hold for a in acts)

    for dt in (6.0, 6.1, 6.2):
        fsm.step(tracker.update(_snap(dt, 1409, 1779)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT
    acts_arr = []
    t = 7.0
    for wp in cfg.active_route.travel_waypoints:
        acts_arr = fsm.step(tracker.update(_snap(t, int(wp.x), int(wp.y))))
        t += 2.0
    assert fsm.state == BotState.SEARCH_MOB
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" and a.reason == "restore_mouse_mode_at_spot" for a in acts_arr)


@pytest.mark.skipif(not SCREENSHOTS_DIR.exists(), reason="Real Vikings Online screenshots not present")
def test_ui_matcher_real_town_and_teleport_screenshots():
    cfg = load_config("config/default.yaml")
    matcher = UiTemplateMatcher(cfg)

    def _load(name: str) -> FramePacket:
        p = SCREENSHOTS_DIR / name
        if not p.exists():
            pytest.skip(f"Missing screenshot {name}")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        return FramePacket(image_bgr=img, timestamp=1.0, source_name=name)

    for fn in (
        "Vikings Online_Mortega-_2026_11_22_13_35(3).jpg",
        "Vikings Online_Mortega-_2026_11_22_13_35(6).jpg",
        "Vikings Online_Mortega-_2026_11_22_13_36(10).jpg",
        "Vikings Online_Mortega-_2026_11_22_13_36(12).jpg",
    ):
        assert matcher.find_addon_keep_sign(_load(fn)) is not None, f"Missed Addon Keep sign in {fn}"

    for fn in (
        "Vikings Online_Mortega-_2026_11_22_13_35(4).jpg",
        "Vikings Online_Mortega-_2026_11_22_13_36(13).jpg",
        "Vikings Online_Mortega-_2026_11_22_13_39(16).jpg",
    ):
        assert matcher.find_addon_keep_sign(_load(fn)) is None, f"False positive Addon Keep sign in {fn}"

    assert matcher.find_teleport_menu(_load("Vikings Online_Mortega-_2026_11_22_13_36(11).jpg")) is not None
    assert matcher.find_teleport_menu(_load("Vikings Online_Mortega-_2026_11_22_13_36(12).jpg")) is not None
    assert matcher.find_teleport_menu(_load("Vikings Online_Mortega-_2026_11_22_13_36(10).jpg")) is None
    assert matcher.find_teleport_menu(_load("Vikings Online_Mortega-_2026_11_22_13_36(13).jpg")) is None


def test_auto_weapon_swap_q_when_ea_low_and_no_swap_when_strong():
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)

    acts_q = fsm.step(tracker.update(_snap(1.5, 615, 1830, ea=94)))
    assert any(isinstance(a, TapKeyIntent) and a.key == "q" for a in acts_q)

    acts_no_q = fsm.step(tracker.update(_snap(5.0, 615, 1830, ea=179)))
    assert not any(isinstance(a, TapKeyIntent) and a.key == "q" for a in acts_no_q)


def test_teleport_dismisses_accidental_npc_talk_dialog_by_clicking_ok_then_enter_fallback():
    cfg = load_config("config/default.yaml", route_override="gold_rustler_spot")
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)

    fsm.resume(1.0, ensure_mouse_mode=False)
    fsm.step(tracker.update(_snap(1.0, 1060, 2626)))
    assert fsm.state == BotState.TELEPORT_TO_SPOT
    fsm.respawn_travel._tele_phase = "WAIT_LOAD"
    fsm.respawn_travel._tele_phase_ts = 2.0

    ok_box = (1026, 553, 55, 25)
    acts1 = fsm.step(tracker.update(_snap(2.2, 1060, 2626, talk_ok_bbox=ok_box)))
    assert fsm.respawn_travel._tele_phase == "FIND_SIGN"
    assert any(
        isinstance(a, ClickScreenIntent) and a.x == 1053 and a.y == 565 and a.no_jitter and "talk_ok" in a.reason
        for a in acts1
    )

    acts2 = fsm.step(tracker.update(_snap(2.75, 1060, 2626, talk_ok_bbox=ok_box)))
    assert any(isinstance(a, TapKeyIntent) and a.key == "enter" for a in acts2)


def test_ui_matcher_detects_npc_talk_dialog_with_zero_false_positives():
    up_dir = Path(r"C:\Users\Desto\.gemini\antigravity\brain\04b1a6fb-5f21-4214-944b-a647347f4420\.user_uploaded")
    talk_shot = up_dir / "media_1795388031935_cdf77952.jpg"
    if not talk_shot.exists():
        pytest.skip("Talk dialog reference screenshot not present")

    cfg = load_config("config/default.yaml")
    matcher = UiTemplateMatcher(cfg)
    assert matcher.ok_tmpl is not None

    im_talk = cv2.imread(str(talk_shot), cv2.IMREAD_COLOR)
    bbox = matcher.find_npc_talk_dialog(FramePacket(image_bgr=im_talk, timestamp=1.0, source_name=talk_shot.name))
    assert bbox is not None

    im_talk_1080 = cv2.resize(im_talk, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    bbox_1080 = matcher.find_npc_talk_dialog(FramePacket(image_bgr=im_talk_1080, timestamp=1.0, source_name="talk_1080"))
    assert bbox_1080 is not None
    assert 1015 <= bbox_1080[0] <= 1040 and 545 <= bbox_1080[1] <= 565

    known_talk_shots = {
        "media_1795388031935_cdf77952.jpg",
        "media_1795609480548_600677a6.jpg",
    }
    for p in sorted(list(up_dir.glob("*.jpg")) + list(up_dir.glob("*.png"))):
        im_other = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im_other is None:
            continue
        fp_box = matcher.find_npc_talk_dialog(FramePacket(image_bgr=im_other, timestamp=1.0, source_name=p.name))
        if p.name in known_talk_shots:
            assert fp_box is not None, f"Expected Talk dialog match on {p.name}"
        else:
            assert fp_box is None, f"False positive Talk dialog on {p.name}: {fp_box}"
