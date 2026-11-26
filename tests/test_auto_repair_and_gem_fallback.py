"""Tests for 80-minute '/kill' Auto-Repair -> Return-to-Spot & Gems 2-Death Spot Fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vikings_bot.brain.fsm import FarmingFSM
from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import load_config
from vikings_bot.input.actions import (
    ActionExecutor,
    ClickScreenIntent,
    HoldKeyIntent,
    TapKeyIntent,
    TypeChatCommandIntent,
)
from vikings_bot.input.driver import DryRunInputDriver
from vikings_bot.perception.ui_matcher import UiTemplateMatcher
from vikings_bot.state.models import (
    BotMode,
    BotState,
    CoordReading,
    MinimapReading,
    PerceptionSnapshot,
)
from vikings_bot.state.tracker import WorldStateTracker
from vikings_bot.targets import build_config_for_target, load_target_profiles


def _make_snap(
    ts: float,
    x: int,
    y: int,
    *,
    fw: int = 1920,
    fh: int = 1080,
    hp: float = 1.0,
    ea: int | None = 180,
    sign_bbox: tuple[int, int, int, int] | None = None,
    menu_bbox: tuple[int, int, int, int] | None = None,
    talk_ok_bbox: tuple[int, int, int, int] | None = None,
    merchant_sign_bbox: tuple[int, int, int, int] | None = None,
    trade_menu_bbox: tuple[int, int, int, int] | None = None,
    repair_all_bbox: tuple[int, int, int, int] | None = None,
    repair_yes_bbox: tuple[int, int, int, int] | None = None,
    inventory_tab_bbox: tuple[int, int, int, int] | None = None,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        timestamp=ts,
        frame_width=fw,
        frame_height=fh,
        hp_ratio=hp,
        mp_ratio=0.9,
        coords=CoordReading(x=x, y=y, confidence=0.99),
        minimap=MinimapReading(heading_deg=0.0, heading_confidence=0.9),
        ea_value=ea,
        addon_keep_sign_bbox=sign_bbox,
        teleport_menu_bbox=menu_bbox,
        npc_talk_dialog_bbox=talk_ok_bbox,
        merchant_sign_bbox=merchant_sign_bbox,
        trade_menu_bbox=trade_menu_bbox,
        repair_all_bbox=repair_all_bbox,
        repair_yes_bbox=repair_yes_bbox,
        inventory_tab_bbox=inventory_tab_bbox,
    )


def _feed_death(tracker: WorldStateTracker, fsm: FarmingFSM, start_ts: float, x: int, y: int, frames: int = 12) -> float:
    t = start_ts
    # Realistic pre-death low HP (<35%) frame before hitting 0% (Anti-False-Death guard)
    fsm.step(tracker.update(_make_snap(t, x, y, hp=0.22)))
    t += 0.05
    for _ in range(frames):
        fsm.step(tracker.update(_make_snap(t, x, y, hp=0.0)))
        t += 0.05
    return t


def test_shop_repair_ui_templates_match_1080p_and_1024x768_with_zero_false_positives() -> None:
    cfg_hd = load_config("config/default.yaml")
    cfg_vm = load_config("config/default.yaml")
    cfg_vm.perception["hud_fixed_pixel_ui"] = True
    cfg_vm.controls["mode"] = "mouse"
    um_hd = UiTemplateMatcher(cfg_hd)
    um_vm = UiTemplateMatcher(cfg_vm)

    assert um_hd.merchant_sign_tmpl is not None
    assert um_hd.trade_menu_tmpl is not None
    assert um_hd.repair_all_tmpl is not None
    assert um_hd.repair_yes_tmpl is not None
    assert um_hd.inv_tab_tmpl is not None

    canvas_hd = np.full((1080, 1920, 3), 45, dtype=np.uint8)
    ms_h, ms_w = um_hd.merchant_sign_tmpl.shape[:2]
    tm_h, tm_w = um_hd.trade_menu_tmpl.shape[:2]
    ra_h, ra_w = um_hd.repair_all_tmpl.shape[:2]
    ry_h, ry_w = um_hd.repair_yes_tmpl.shape[:2]
    it_h, it_w = um_hd.inv_tab_tmpl.shape[:2]

    canvas_hd[472 : 472 + ms_h, 872 : 872 + ms_w] = um_hd.merchant_sign_tmpl
    canvas_hd[328 : 328 + tm_h, 1004 : 1004 + tm_w] = um_hd.trade_menu_tmpl
    canvas_hd[532 : 532 + ra_h, 1492 : 1492 + ra_w] = um_hd.repair_all_tmpl
    canvas_hd[146 : 146 + it_h, 1590 : 1590 + it_w] = um_hd.inv_tab_tmpl

    canvas_yes_hd = np.full((1080, 1920, 3), 45, dtype=np.uint8)
    canvas_yes_hd[480:572, 835:1088] = 18
    canvas_yes_hd[538 : 538 + ry_h, 964 : 964 + ry_w] = um_hd.repair_yes_tmpl

    fp_hd = FramePacket(image_bgr=canvas_hd, timestamp=1.0)
    fp_yes_hd = FramePacket(image_bgr=canvas_yes_hd, timestamp=1.1)
    assert um_hd.find_merchant_sign(fp_hd) == (872, 472, ms_w, ms_h)
    assert um_hd.find_trade_menu(fp_hd) == (1004, 328, tm_w, tm_h)
    assert um_hd.find_repair_all_btn(fp_hd) == (1492, 532, ra_w, ra_h)
    assert um_hd.find_repair_yes_btn(fp_yes_hd) == (964, 538, ry_w, ry_h)
    assert um_hd.find_inventory_tab(fp_hd) == (1590, 146, it_w, it_h)

    canvas_vm = np.full((768, 1024, 3), 45, dtype=np.uint8)
    canvas_vm[360 : 360 + ms_h, 420 : 420 + ms_w] = um_vm.merchant_sign_tmpl
    canvas_vm[245 : 245 + tm_h, 510 : 510 + tm_w] = um_vm.trade_menu_tmpl
    canvas_vm[532 : 532 + ra_h, 596 : 596 + ra_w] = um_vm.repair_all_tmpl
    canvas_vm[146 : 146 + it_h, 694 : 694 + it_w] = um_vm.inv_tab_tmpl

    canvas_yes_vm = np.full((768, 1024, 3), 45, dtype=np.uint8)
    canvas_yes_vm[380:472, 390:640] = 18
    canvas_yes_vm[438 : 438 + ry_h, 510 : 510 + ry_w] = um_vm.repair_yes_tmpl

    fp_vm = FramePacket(image_bgr=canvas_vm, timestamp=2.0)
    fp_yes_vm = FramePacket(image_bgr=canvas_yes_vm, timestamp=2.1)
    assert um_vm.find_merchant_sign(fp_vm) == (420, 360, ms_w, ms_h)
    assert um_vm.find_trade_menu(fp_vm) == (510, 245, tm_w, tm_h)
    assert um_vm.find_repair_all_btn(fp_vm) == (596, 532, ra_w, ra_h)
    assert um_vm.find_repair_yes_btn(fp_yes_vm) == (510, 438, ry_w, ry_h)
    assert um_vm.find_inventory_tab(fp_vm) == (694, 146, it_w, it_h)

    blank = FramePacket(image_bgr=np.full((1080, 1920, 3), 55, dtype=np.uint8), timestamp=3.0)
    assert um_hd.find_merchant_sign(blank) is None
    assert um_hd.find_trade_menu(blank) is None
    assert um_hd.find_repair_all_btn(blank) is None
    assert um_hd.find_repair_yes_btn(blank) is None
    assert um_hd.find_inventory_tab(blank) is None

    ss_dir = Path(r"C:\Program Files (x86)\Vikings Online\ScreenShots")
    s9_p = ss_dir / "Vikings Online_Silence-_2026_11_25_13_21(9).jpg"
    s10_p = ss_dir / "Vikings Online_Silence-_2026_11_25_13_21(10).jpg"
    s12_p = ss_dir / "Vikings Online_Silence-_2026_11_25_13_21(12).jpg"
    if s9_p.exists() and s10_p.exists() and s12_p.exists():
        im9 = cv2.imread(str(s9_p))
        im10 = cv2.imread(str(s10_p))
        im12 = cv2.imread(str(s12_p))
        assert um_hd.find_repair_all_btn(FramePacket(im9, 0.0)) == (1492, 532, 66, 25)
        assert um_hd.find_inventory_tab(FramePacket(im9, 0.0)) == (1590, 146, 105, 22)
        assert um_hd.find_repair_yes_btn(FramePacket(im10, 0.0)) == (964, 538, 55, 25)
        assert um_hd.find_repair_all_btn(FramePacket(im12, 0.0)) is None
        assert um_hd.find_repair_yes_btn(FramePacket(im12, 0.0)) is None


def test_full_80m_auto_repair_cycle_with_weapon_swap_talk_guard_and_return_to_spot() -> None:
    profs = {p.id: p for p in load_target_profiles("config/targets.yaml")}
    cfg = build_config_for_target(profs["gems"], "config/default.yaml", control_mode="keyboard", humanize=False)
    cfg.respawn_travel["mouse_only_travel"] = False
    tracker = WorldStateTracker(death_zero_hp_frames=12)
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    drv = DryRunInputDriver()
    exe = ActionExecutor(driver=drv, tracker=tracker, click_jitter_px=0)

    fsm.resume(1.0)
    fsm.step(tracker.update(_make_snap(1.0, 1195, 1568)))
    assert fsm.state == BotState.SEARCH_MOB

    # Fast-forward active farming timer to 80 minutes (4800s) -> swaps weapon Q if EA < 130, then enters REPAIR_RECALL
    fsm._active_farming_sec = 4801.0
    w_low_ea = tracker.update(_make_snap(5.2, 1195, 1568, ea=95))
    acts_swap = fsm.step(w_low_ea)
    assert any(isinstance(a, TapKeyIntent) and a.key == "q" for a in acts_swap)
    assert fsm.state == BotState.SEARCH_MOB

    w_main_ea = tracker.update(_make_snap(5.6, 1195, 1568, ea=180))
    fsm.step(w_main_ea)
    assert fsm.state == BotState.REPAIR_RECALL
    assert fsm.deaths_count == 0

    acts_kill = fsm.step(tracker.update(_make_snap(5.8, 1195, 1568)))
    assert any(isinstance(a, TypeChatCommandIntent) and a.command == "/kill" for a in acts_kill)
    exe.execute_many(acts_kill, 5.8)
    assert "send_chat_command(/kill)" in drv.history

    w_town = tracker.update(_make_snap(22.5, 1075, 2628))
    fsm.step(w_town)
    assert fsm.state == BotState.REPAIR_AT_NPC
    assert fsm.respawn_travel._repair_phase == "WALK_TO_MERCHANT"

    fsm.step(tracker.update(_make_snap(25.0, 1094, 2643)))
    assert fsm.respawn_travel._repair_phase == "FIND_MERCHANT_SIGN"

    ms_box = (872, 472, 190, 27)
    acts_esc = fsm.step(tracker.update(_make_snap(25.2, 1094, 2643, merchant_sign_bbox=ms_box)))
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" for a in acts_esc)
    acts_npc = fsm.step(tracker.update(_make_snap(25.5, 1094, 2643, merchant_sign_bbox=ms_box)))
    assert any(isinstance(a, ClickScreenIntent) and a.button == "left" for a in acts_npc)
    assert fsm.respawn_travel._repair_phase == "WAIT_TRADE_MENU"

    talk_box = (1026, 568, 55, 25)
    acts_tok1 = fsm.step(tracker.update(_make_snap(25.8, 1094, 2643, talk_ok_bbox=talk_box)))
    assert any(isinstance(a, ClickScreenIntent) and a.x == 1053 and a.y == 580 for a in acts_tok1)
    acts_tok2 = fsm.step(tracker.update(_make_snap(26.4, 1094, 2643, talk_ok_bbox=talk_box)))
    assert any(isinstance(a, TapKeyIntent) and a.key == "enter" for a in acts_tok2)
    assert fsm.respawn_travel._repair_phase == "FIND_MERCHANT_SIGN"

    tm_box = (1004, 328, 58, 43)
    acts_tr = fsm.step(tracker.update(_make_snap(27.5, 1094, 2643, trade_menu_bbox=tm_box)))
    if not acts_tr or not any(isinstance(a, ClickScreenIntent) for a in acts_tr):
        acts_tr = fsm.step(tracker.update(_make_snap(27.6, 1094, 2643, trade_menu_bbox=tm_box)))
    assert any(isinstance(a, ClickScreenIntent) and "click_trade_option" in a.reason for a in acts_tr)
    assert fsm.respawn_travel._repair_phase == "CLICK_REPAIR_ALL"

    ra_box = (1492, 532, 66, 25)
    it_box = (1590, 146, 105, 22)
    acts_ra = fsm.step(
        tracker.update(_make_snap(28.0, 1094, 2643, repair_all_bbox=ra_box, inventory_tab_bbox=it_box))
    )
    assert any(
        isinstance(a, ClickScreenIntent) and a.x == 1525 and a.y == 544 and "click_repair_all_btn" in a.reason
        for a in acts_ra
    )
    assert fsm.respawn_travel._repair_phase == "CONFIRM_REPAIR"

    yes_box = (964, 538, 55, 25)
    acts_yes = fsm.step(
        tracker.update(
            _make_snap(28.4, 1094, 2643, repair_all_bbox=ra_box, repair_yes_bbox=yes_box, inventory_tab_bbox=it_box)
        )
    )
    assert any(
        isinstance(a, ClickScreenIntent) and a.x == 991 and a.y == 550 and "click_repair_yes_btn" in a.reason
        for a in acts_yes
    )
    assert fsm.respawn_travel._repair_phase == "CLOSE_SHOP_INV"

    acts_cs = fsm.step(
        tracker.update(_make_snap(28.9, 1094, 2643, repair_all_bbox=ra_box, inventory_tab_bbox=it_box))
    )
    assert any(
        isinstance(a, ClickScreenIntent) and a.x == 1578 and a.y == 140 and "click_shop_close_x" in a.reason
        for a in acts_cs
    )

    acts_ci = fsm.step(tracker.update(_make_snap(29.4, 1094, 2643, inventory_tab_bbox=it_box)))
    assert any(
        isinstance(a, ClickScreenIntent) and a.x == 1913 and a.y == 140 and "click_inventory_close_x" in a.reason
        for a in acts_ci
    )

    acts_done = fsm.step(tracker.update(_make_snap(29.9, 1094, 2643)))
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" and "hide_cursor_after_repair" in a.reason for a in acts_done)
    assert fsm.respawn_travel._repair_phase == "WALK_TO_HUB"

    fsm.step(tracker.update(_make_snap(31.5, 1075, 2631)))
    assert fsm.state == BotState.TELEPORT_TO_SPOT
    assert fsm.repairs_count == 1
    assert fsm._active_farming_sec == 0.0
    assert fsm.deaths_count == 0

    fsm.step(tracker.update(_make_snap(38.0, 1409, 1779)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT
    ts = 39.0
    for wp in cfg.active_route.travel_waypoints:
        fsm.step(tracker.update(_make_snap(ts, int(wp.x), int(wp.y))))
        ts += 2.0
    assert fsm.state == BotState.SEARCH_MOB
    assert abs(fsm.navigator.spot_center.x - 1195.0) <= 1.0
    assert abs(fsm.navigator.spot_center.y - 1568.0) <= 1.0


def test_1024x768_pure_mouse_auto_repair_never_uses_wasd_or_esc() -> None:
    cfg = load_config("config/default.yaml")
    cfg.controls["mode"] = "mouse"
    cfg.controls["auto_restore_mouse_mode"] = True
    cfg.respawn_travel["mouse_only_travel"] = True
    cfg.perception["hud_fixed_pixel_ui"] = True
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0)
    fsm._mouse_mode_verified = True
    fsm.step(tracker.update(_make_snap(1.0, 615, 1830, fw=1024, fh=768)))

    fsm.request_repair_now()
    fsm.step(tracker.update(_make_snap(1.2, 615, 1830, fw=1024, fh=768)))
    assert fsm.state == BotState.REPAIR_RECALL

    all_intents = []
    all_intents.extend(fsm.step(tracker.update(_make_snap(1.4, 615, 1830, fw=1024, fh=768))))
    all_intents.extend(fsm.step(tracker.update(_make_snap(19.0, 1075, 2628, fw=1024, fh=768))))
    assert fsm.state == BotState.REPAIR_AT_NPC

    all_intents.extend(fsm.step(tracker.update(_make_snap(19.5, 1075, 2628, fw=1024, fh=768))))
    all_intents.extend(fsm.step(tracker.update(_make_snap(21.0, 1094, 2643, fw=1024, fh=768))))
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(21.3, 1094, 2643, fw=1024, fh=768, merchant_sign_bbox=(420, 360, 190, 27))))
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(21.7, 1094, 2643, fw=1024, fh=768, trade_menu_bbox=(510, 245, 58, 43))))
    )
    all_intents.extend(
        fsm.step(
            tracker.update(
                _make_snap(
                    22.1, 1094, 2643, fw=1024, fh=768, repair_all_bbox=(596, 532, 66, 25), inventory_tab_bbox=(694, 146, 105, 22)
                )
            )
        )
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(22.5, 1094, 2643, fw=1024, fh=768, repair_yes_bbox=(510, 438, 55, 25))))
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(23.0, 1094, 2643, fw=1024, fh=768, repair_all_bbox=(596, 532, 66, 25))))
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(23.5, 1094, 2643, fw=1024, fh=768, inventory_tab_bbox=(694, 146, 105, 22))))
    )
    all_intents.extend(fsm.step(tracker.update(_make_snap(24.0, 1094, 2643, fw=1024, fh=768))))
    all_intents.extend(fsm.step(tracker.update(_make_snap(25.5, 1075, 2631, fw=1024, fh=768))))

    assert fsm.state == BotState.TELEPORT_TO_SPOT
    assert not any(isinstance(a, HoldKeyIntent) and a.hold and a.key in ("w", "a", "s", "d") for a in all_intents)
    assert not any(isinstance(a, TapKeyIntent) and a.key == "esc" for a in all_intents)


def test_gems_new_spot_1195_1568_switches_to_old_spot_1092_1115_only_after_2_real_deaths() -> None:
    profs = {p.id: p for p in load_target_profiles("config/targets.yaml")}
    cfg = build_config_for_target(profs["gems"], "config/default.yaml", control_mode="keyboard", humanize=False)
    cfg.respawn_travel["mouse_only_travel"] = False
    tracker = WorldStateTracker(death_zero_hp_frames=6)
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)

    assert cfg.active_route_name == "gem_rustler_spot"
    assert (fsm.configured_spot_center.x, fsm.configured_spot_center.y) == (1195.0, 1568.0)

    fsm.resume(1.0)
    fsm.step(tracker.update(_make_snap(1.0, 1195, 1568)))
    assert fsm.state == BotState.SEARCH_MOB
    assert (fsm.navigator.spot_center.x, fsm.navigator.spot_center.y) == (1195.0, 1568.0)

    fsm.request_repair_now()
    fsm.step(tracker.update(_make_snap(2.0, 1195, 1568)))
    assert fsm.state == BotState.REPAIR_RECALL
    assert fsm.deaths_count == 0
    assert cfg.active_route_name == "gem_rustler_spot"
    fsm.transition(BotState.SEARCH_MOB, 3.0, "resume_test")

    t = _feed_death(tracker, fsm, 4.0, 1195, 1568, frames=6)
    assert fsm.state == BotState.DEAD_RESPAWN
    assert fsm.deaths_count == 1
    assert cfg.active_route_name == "gem_rustler_spot"
    assert (fsm.configured_spot_center.x, fsm.configured_spot_center.y) == (1195.0, 1568.0)

    fsm.transition(BotState.SEARCH_MOB, t + 5.0, "back_at_primary")
    fsm.navigator.set_dynamic_anchor(fsm.configured_spot_center)

    t2 = _feed_death(tracker, fsm, t + 6.0, 1195, 1568, frames=6)
    assert fsm.state == BotState.DEAD_RESPAWN
    assert fsm.deaths_count == 2
    assert cfg.active_route_name == "gem_rustler_fallback_spot"
    assert (fsm.configured_spot_center.x, fsm.configured_spot_center.y) == (1092.0, 1115.0)
    assert (fsm.navigator.spot_center.x, fsm.navigator.spot_center.y) == (1092.0, 1115.0)

    fsm.step(tracker.update(_make_snap(t2 + 3.0, 1075, 2628)))
    assert fsm.state == BotState.TELEPORT_TO_SPOT
    fsm.step(tracker.update(_make_snap(t2 + 10.0, 1409, 1779)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT
    ts = t2 + 11.0
    for wp in cfg.active_route.travel_waypoints:
        fsm.step(tracker.update(_make_snap(ts, int(wp.x), int(wp.y))))
        ts += 2.0
    assert fsm.state == BotState.SEARCH_MOB
    assert abs(fsm.navigator.spot_center.x - 1092.0) <= 1.0
    assert abs(fsm.navigator.spot_center.y - 1115.0) <= 1.0
