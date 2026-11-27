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

    acts_kill = fsm.step(tracker.update(_make_snap(5.9, 1195, 1568)))
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
    acts_tab = fsm.step(tracker.update(_make_snap(25.2, 1094, 2643, merchant_sign_bbox=ms_box)))
    assert any(isinstance(a, TapKeyIntent) and a.key == "tab" for a in acts_tab)
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
    assert any(isinstance(a, TapKeyIntent) and a.key == "tab" and "hide_cursor_after_repair" in a.reason for a in acts_done)
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


def test_1024x768_keyboard_mode_auto_repair_uses_wasd_and_tab_cursor_never_esc_or_ground_click() -> None:
    cfg = load_config("config/default.yaml")
    assert cfg.respawn_travel["mouse_only_travel"] is False
    cfg.perception["hud_fixed_pixel_ui"] = True
    cfg.repair["close_dialog_with_esc"] = False
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    fsm._mouse_mode_verified = True
    fsm.step(tracker.update(_make_snap(1.0, 627, 1817, fw=1024, fh=768)))

    fsm.request_repair_now()
    fsm.step(tracker.update(_make_snap(1.2, 627, 1817, fw=1024, fh=768)))
    assert fsm.state == BotState.REPAIR_RECALL

    all_intents = []
    walk_intents = []
    all_intents.extend(fsm.step(tracker.update(_make_snap(1.4, 627, 1817, fw=1024, fh=768))))
    all_intents.extend(fsm.step(tracker.update(_make_snap(19.0, 1075, 2631, fw=1024, fh=768))))
    assert fsm.state == BotState.REPAIR_AT_NPC

    # Walking to merchant (1075,2631 -> 1094,2644): MUST use Keyboard WASD ('w'), NEVER ground clicks or Esc!
    w_step = fsm.step(tracker.update(_make_snap(19.5, 1075, 2631, fw=1024, fh=768)))
    walk_intents.extend(w_step)
    all_intents.extend(w_step)
    all_intents.extend(fsm.step(tracker.update(_make_snap(21.0, 1094, 2644, fw=1024, fh=768))))
    # Show Tab cursor then click NPC -> Trade -> Repair All -> Yes -> Close [x] -> Hide Tab cursor
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(21.3, 1094, 2644, fw=1024, fh=768, merchant_sign_bbox=(420, 360, 190, 27))))
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(21.55, 1094, 2644, fw=1024, fh=768, merchant_sign_bbox=(420, 360, 190, 27))))
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(21.9, 1094, 2644, fw=1024, fh=768, trade_menu_bbox=(510, 245, 58, 43))))
    )
    all_intents.extend(
        fsm.step(
            tracker.update(
                _make_snap(
                    22.3, 1094, 2644, fw=1024, fh=768, repair_all_bbox=(596, 532, 66, 25), inventory_tab_bbox=(694, 146, 105, 22)
                )
            )
        )
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(22.7, 1094, 2644, fw=1024, fh=768, repair_yes_bbox=(510, 438, 55, 25))))
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(23.2, 1094, 2644, fw=1024, fh=768, repair_all_bbox=(596, 532, 66, 25))))
    )
    all_intents.extend(
        fsm.step(tracker.update(_make_snap(23.7, 1094, 2644, fw=1024, fh=768, inventory_tab_bbox=(694, 146, 105, 22))))
    )
    all_intents.extend(fsm.step(tracker.update(_make_snap(24.2, 1094, 2644, fw=1024, fh=768))))
    hub_walk = fsm.step(tracker.update(_make_snap(24.5, 1086, 2639, fw=1024, fh=768)))
    walk_intents.extend(hub_walk)
    all_intents.extend(hub_walk)
    all_intents.extend(fsm.step(tracker.update(_make_snap(25.5, 1075, 2631, fw=1024, fh=768))))

    assert fsm.state == BotState.TELEPORT_TO_SPOT
    assert any(isinstance(a, HoldKeyIntent) and a.hold and a.key == "w" for a in walk_intents)
    assert not any(isinstance(a, ClickScreenIntent) for a in walk_intents)
    assert any(isinstance(a, TapKeyIntent) and a.key == "tab" for a in all_intents)
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


def test_f7_force_repair_hotkey_100ms_clicks_esc_close_and_gold_rustler_594_1832_return() -> None:
    """Verify F7 Force Repair hotkey, 210ms AHK /kill interlock, 100ms NPC/UI click-hold,
    both popup branches ('Yes' and 'No equipment to repair.') + Talk OK recovery, Tab cursor,
    city auto-detect on F9, Addon Keep Teleport (1066, 2627) -> Ardun Keep (1409, 1773) -> (627, 1817) + /setweather 11."""
    from vikings_bot.runtime.safety import SafetyMonitor

    cfg = load_config("config/default.yaml")
    cfg.perception["hud_fixed_pixel_ui"] = True
    assert str(cfg.safety.get("force_repair_key")).upper() == "F7"
    assert float(cfg.repair.get("npc_click_hold_sec", 0.0)) == 0.10
    assert bool(cfg.repair.get("close_dialog_with_esc", True)) is False
    assert str(cfg.respawn_travel.get("cursor_toggle_key")).lower() == "tab"
    assert cfg.farming["anchor_spot_on_resume"] is False

    # 1. AHK v2 bridge hard interlock (>=210ms sleep on CHAT_CMD, 40ms settle before CLICK)
    ahk_src = Path("ahk/vikings_input_bridge.ahk").read_text(encoding="utf-8")
    assert "Sleep 210" in ahk_src and "Sleep 40" in ahk_src

    # 2. SafetyMonitor F7 hotkey triggers force_repair_requested
    sm = SafetyMonitor(cfg, enable_hotkeys=True)
    assert sm.hotkeys is not None and "force_repair" in sm.hotkeys._keys
    sm.hotkeys.poll = lambda: {"force_repair"}  # type: ignore[method-assign]
    chk = sm.check(None, is_running_active=True)
    assert chk.force_repair_requested is True

    # 3. UI template matcher records best match scores on miss & handles optional missing PNG gracefully
    um = UiTemplateMatcher(cfg)
    blank = FramePacket(image_bgr=np.full((768, 1024, 3), 50, dtype=np.uint8), timestamp=10.0)
    assert um.find_merchant_sign(blank) is None
    assert "merchant_sign" in um.last_match_scores
    assert um.find_no_equipment_repair(blank) is None

    # 4. City auto-detect on F9 (x>=1000, y>=2500) -> leaves town directly instead of SCAN_SPOT
    tr_city = WorldStateTracker()
    fsm_city = FarmingFSM(cfg, tr_city, mode=BotMode.FULL)
    fsm_city.resume(1.0, ensure_mouse_mode=False)
    fsm_city.step(tr_city.update(_make_snap(1.0, 1066, 2627, fw=1024, fh=768)))
    assert fsm_city.state == BotState.TELEPORT_TO_SPOT

    # 5. Full F7 Force Repair -> /kill -> REPAIR_AT_NPC (100ms clicks, Yes branch + Talk OK recovery) -> Ardun Keep -> (627, 1817)
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    assert (fsm.configured_spot_center.x, fsm.configured_spot_center.y) == (627.0, 1817.0)
    fsm.resume(1.0, ensure_mouse_mode=False)
    fsm.step(tracker.update(_make_snap(1.0, 627, 1817, fw=1024, fh=768)))
    # Pause does NOT reset active farming timer
    fsm._active_farming_sec = 1234.0
    fsm.pause(2.0, "test_pause")
    assert fsm._active_farming_sec == 1234.0
    fsm.resume(3.0, ensure_mouse_mode=False)
    assert fsm._active_farming_sec == 1234.0

    # F7 Force Repair -> repair_rem_min == 0 -> enters REPAIR_RECALL immediately!
    fsm.request_repair_now()
    assert fsm.repair_remaining_minutes == 0.0
    fsm.step(tracker.update(_make_snap(3.5, 627, 1817, fw=1024, fh=768)))
    assert fsm.state == BotState.REPAIR_RECALL

    acts_k = fsm.step(tracker.update(_make_snap(3.8, 627, 1817, fw=1024, fh=768)))
    assert any(isinstance(a, TypeChatCommandIntent) and a.command == "/kill" for a in acts_k)

    # Arrives in Green Wales (1075, 2631) -> walks to merchant (1094, 2644) -> Tab cursor -> 100ms clicks + Talk OK recovery + Yes!
    fsm.step(tracker.update(_make_snap(21.0, 1075, 2631, fw=1024, fh=768)))
    assert fsm.state == BotState.REPAIR_AT_NPC
    fsm.step(tracker.update(_make_snap(23.0, 1094, 2644, fw=1024, fh=768)))
    a_tab = fsm.step(tracker.update(_make_snap(23.1, 1094, 2644, fw=1024, fh=768, merchant_sign_bbox=(420, 360, 190, 27))))
    assert any(isinstance(a, TapKeyIntent) and a.key == "tab" for a in a_tab)
    a_npc = fsm.step(tracker.update(_make_snap(23.35, 1094, 2644, fw=1024, fh=768, merchant_sign_bbox=(420, 360, 190, 27))))
    assert any(isinstance(a, ClickScreenIntent) and abs(a.hold_sec - 0.10) < 1e-4 for a in a_npc)
    # Accidental Talk OK recovery
    a_tok = fsm.step(tracker.update(_make_snap(23.5, 1094, 2644, fw=1024, fh=768, talk_ok_bbox=(480, 400, 55, 25))))
    assert any(isinstance(a, ClickScreenIntent) and "click_merchant_talk_ok" in a.reason for a in a_tok)
    # Trade -> Repair All -> Yes
    a_tr = fsm.step(tracker.update(_make_snap(23.9, 1094, 2644, fw=1024, fh=768, trade_menu_bbox=(510, 245, 58, 43))))
    if not any(isinstance(a, ClickScreenIntent) for a in a_tr):
        a_tr = fsm.step(tracker.update(_make_snap(24.0, 1094, 2644, fw=1024, fh=768, trade_menu_bbox=(510, 245, 58, 43))))
    assert any(isinstance(a, ClickScreenIntent) and abs(a.hold_sec - 0.10) < 1e-4 for a in a_tr)
    a_ra = fsm.step(tracker.update(_make_snap(24.4, 1094, 2644, fw=1024, fh=768, repair_all_bbox=(596, 532, 66, 25))))
    assert any(isinstance(a, ClickScreenIntent) and abs(a.hold_sec - 0.10) < 1e-4 for a in a_ra)
    a_yes = fsm.step(tracker.update(_make_snap(24.8, 1094, 2644, fw=1024, fh=768, repair_yes_bbox=(510, 438, 55, 25))))
    assert any(isinstance(a, ClickScreenIntent) and abs(a.hold_sec - 0.10) < 1e-4 and "click_repair_yes_btn" in a.reason for a in a_yes)

    # Branch 2 verification: 'No equipment to repair.' in CONFIRM_REPAIR immediately advances to CLOSE_SHOP_INV!
    fsm.respawn_travel._repair_phase = "CONFIRM_REPAIR"
    snap_no_eq = PerceptionSnapshot(
        timestamp=25.0, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
        coords=CoordReading(x=1094, y=2644, confidence=0.99), minimap=MinimapReading(),
        no_equipment_repair_bbox=(420, 360, 180, 28),
    )
    fsm.step(tracker.update(snap_no_eq))
    assert fsm.respawn_travel._repair_phase == "CLOSE_SHOP_INV"

    # Dialog closed -> walks to hub (1075, 2631) -> TELEPORT_TO_SPOT -> resets 80m timer!
    fsm.step(tracker.update(_make_snap(25.5, 1094, 2644, fw=1024, fh=768)))
    fsm.step(tracker.update(_make_snap(27.0, 1075, 2631, fw=1024, fh=768)))
    assert fsm.state == BotState.TELEPORT_TO_SPOT
    assert fsm.repairs_count == 1 and fsm._active_farming_sec == 0.0

    # Teleports to Ardun Keep (1409, 1773) -> TRAVEL_TO_SPOT -> traverses travel_waypoints -> arrives (627, 1817)!
    fsm.step(tracker.update(_make_snap(32.0, 1409, 1773, fw=1024, fh=768)))
    assert fsm.state == BotState.TRAVEL_TO_SPOT
    ts = 33.0
    for wp in cfg.active_route.travel_waypoints:
        fsm.step(tracker.update(_make_snap(ts, int(wp.x), int(wp.y), fw=1024, fh=768)))
        ts += 1.5
    assert fsm.state == BotState.SEARCH_MOB
    assert (fsm.navigator.spot_center.x, fsm.navigator.spot_center.y) == (627.0, 1817.0)

    # First SEARCH_MOB tick back at (627, 1817) sends /setweather 11!
    w_back = tracker.update(
        PerceptionSnapshot(
            timestamp=ts + 2.0, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
            coords=CoordReading(x=627, y=1817, confidence=0.99), minimap=MinimapReading(),
            playfield_luma=90.0,
        )
    )
    acts_w = fsm.step(w_back)
    assert any(isinstance(a, TypeChatCommandIntent) and a.command == "/setweather 11" for a in acts_w)


def test_ensure_keyboard_mode_transitions_kill_freeze_and_no_esc_after_repair_yes() -> None:
    """Verify:
    (a) REPAIR_RECALL freezes character completely for 0.25s before /kill and NEVER sends clicks/moves/WASD/Esc during 16s countdown.
    (b) REPAIR_RECALL -> city and F9 resume in Green Wales / Ardun Keep trigger ENSURE_KEYBOARD_MODE (2x Esc if mouse_banner, 1x Esc if keyboard_banner).
    (c) step_repair_at_npc after clicking Yes NEVER sends Esc (even if close_dialog_with_esc=True).
    (d) TRAVEL_TO_SPOT arrival at (627, 1817) triggers ENSURE_MOUSE_MODE before SEARCH_MOB.
    """
    from vikings_bot.input.actions import MoveMouseRelIntent

    cfg = load_config("config/default.yaml")
    cfg.perception["hud_fixed_pixel_ui"] = True

    # --- (b1) F9 resume in Green Wales town -> redirects directly to ENSURE_KEYBOARD_MODE (2x Esc on mouse_banner) -> TELEPORT_TO_SPOT ---
    tr_town = WorldStateTracker()
    fsm_town = FarmingFSM(cfg, tr_town, mode=BotMode.FULL)
    fsm_town.resume(1.0)  # default ensure_mouse_mode=None (real runtime)
    assert fsm_town.state == BotState.ENSURE_MOUSE_MODE
    a_probe1 = fsm_town.step(tr_town.update(_make_snap(1.0, 1066, 2627, fw=1024, fh=768)))
    assert fsm_town.state == BotState.ENSURE_KEYBOARD_MODE
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" and a.reason == "ensure_keyboard_mode_esc_1" for a in a_probe1)
    # After 0.32s, banner shows mouse_banner (we were already in Keyboard Mode -> flipped to Mouse) -> sends 2nd Esc!
    sn_mb = PerceptionSnapshot(
        timestamp=1.35, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
        coords=CoordReading(x=1066, y=2627, confidence=0.99), minimap=MinimapReading(),
        interface_mode="mouse_banner",
    )
    a_probe2 = fsm_town.step(tr_town.update(sn_mb))
    assert fsm_town.state == BotState.ENSURE_KEYBOARD_MODE
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" and a.reason == "ensure_keyboard_mode_esc_2" for a in a_probe2)
    # After 0.28s -> enters TELEPORT_TO_SPOT!
    fsm_town.step(tr_town.update(_make_snap(1.66, 1066, 2627, fw=1024, fh=768)))
    assert fsm_town.state == BotState.TELEPORT_TO_SPOT

    # --- (b2) F9 resume at Ardun Keep (1409, 1773) -> ENSURE_KEYBOARD_MODE (1x Esc on keyboard_banner) -> TRAVEL_TO_SPOT ---
    tr_ak = WorldStateTracker()
    fsm_ak = FarmingFSM(cfg, tr_ak, mode=BotMode.FULL)
    fsm_ak.resume(1.0)
    a_ak1 = fsm_ak.step(tr_ak.update(_make_snap(1.0, 1409, 1773, fw=1024, fh=768)))
    assert fsm_ak.state == BotState.ENSURE_KEYBOARD_MODE
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" and a.reason == "ensure_keyboard_mode_esc_1" for a in a_ak1)
    sn_kb = PerceptionSnapshot(
        timestamp=1.35, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
        coords=CoordReading(x=1409, y=1773, confidence=0.99), minimap=MinimapReading(),
        interface_mode="keyboard_banner",
    )
    a_ak2 = fsm_ak.step(tr_ak.update(sn_kb))
    assert a_ak2 == []
    assert fsm_ak.state == BotState.TRAVEL_TO_SPOT

    # --- (a) + (b3) + (c) + (d): Full cycle REPAIR_RECALL (frozen /kill) -> ENSURE_KEYBOARD_MODE -> REPAIR_AT_NPC (no Esc after Yes) -> TRAVEL_TO_SPOT -> ENSURE_MOUSE_MODE -> SEARCH_MOB ---
    tr = WorldStateTracker()
    fsm = FarmingFSM(cfg, tr, mode=BotMode.FULL)
    fsm.respawn_travel.close_dialog_with_esc = True  # Even if legacy flag set, MUST NOT send Esc after Yes!
    fsm.resume(1.0)
    # Finish initial ENSURE_MOUSE_MODE at farming spot (627, 1817)
    fsm.step(tr.update(_make_snap(1.0, 627, 1817, fw=1024, fh=768)))
    fsm.step(
        tr.update(
            PerceptionSnapshot(
                timestamp=1.35, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
                coords=CoordReading(x=627, y=1817, confidence=0.99), minimap=MinimapReading(),
                interface_mode="mouse_banner",
            )
        )
    )
    assert fsm.state == BotState.SEARCH_MOB

    # Trigger F7 Force Repair -> enters REPAIR_RECALL at t=2.0
    fsm.request_repair_now()
    fsm.step(tr.update(_make_snap(2.0, 627, 1817, fw=1024, fh=768)))
    assert fsm.state == BotState.REPAIR_RECALL

    # 0.10s after entry (< 0.25s settle): frozen, does NOT send /kill yet!
    a_pre = fsm.step(tr.update(_make_snap(2.10, 627, 1817, fw=1024, fh=768)))
    assert not any(isinstance(a, TypeChatCommandIntent) for a in a_pre)

    # 0.30s after entry (>= 0.25s): sends /kill once, then stands completely still through 16s countdown
    recall_acts = list(a_pre)
    for ts in (2.30, 4.0, 8.0, 12.0, 16.0):
        # Even if respawn_dialog_bbox flickers during countdown, NEVER click screen!
        sn_cd = PerceptionSnapshot(
            timestamp=ts, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
            coords=CoordReading(x=627, y=1817, confidence=0.99), minimap=MinimapReading(),
            respawn_dialog_bbox=(480, 400, 60, 28) if ts == 8.0 else None,
        )
        recall_acts.extend(fsm.step(tr.update(sn_cd)))
    assert any(isinstance(a, TypeChatCommandIntent) and a.command == "/kill" for a in recall_acts)
    assert not any(isinstance(a, ClickScreenIntent) for a in recall_acts)
    assert not any(isinstance(a, MoveMouseRelIntent) for a in recall_acts)
    assert not any(isinstance(a, HoldKeyIntent) and a.hold for a in recall_acts)
    assert not any(isinstance(a, TapKeyIntent) and a.key == "esc" for a in recall_acts)

    # Respawned in Green Wales (1075, 2631) -> MUST enter ENSURE_KEYBOARD_MODE first!
    fsm.step(tr.update(_make_snap(19.0, 1075, 2631, fw=1024, fh=768)))
    assert fsm.state == BotState.ENSURE_KEYBOARD_MODE
    fsm.step(tr.update(_make_snap(19.05, 1075, 2631, fw=1024, fh=768)))  # sends Esc #1
    fsm.step(
        tr.update(
            PerceptionSnapshot(
                timestamp=19.40, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
                coords=CoordReading(x=1075, y=2631, confidence=0.99), minimap=MinimapReading(),
                interface_mode="keyboard_banner",
            )
        )
    )
    assert fsm.state == BotState.REPAIR_AT_NPC

    # (c) Clicking Yes in REPAIR_AT_NPC NEVER sends Esc!
    fsm.respawn_travel._repair_phase = "CONFIRM_REPAIR"
    a_yes = fsm.step(tr.update(_make_snap(22.0, 1094, 2644, fw=1024, fh=768, repair_yes_bbox=(516, 382, 55, 25))))
    assert any(isinstance(a, ClickScreenIntent) and a.reason == "click_repair_yes_btn" for a in a_yes)
    assert not any(isinstance(a, TapKeyIntent) and a.key == "esc" for a in a_yes)

    # (d) TRAVEL_TO_SPOT arrives at (627, 1817) -> MUST trigger ENSURE_MOUSE_MODE before SEARCH_MOB!
    fsm.transition(BotState.TRAVEL_TO_SPOT, 30.0, "test_travel_arrival")
    fsm.step(tr.update(_make_snap(31.0, 627, 1817, fw=1024, fh=768)))
    assert fsm.state == BotState.ENSURE_MOUSE_MODE
    a_m1 = fsm.step(tr.update(_make_snap(31.05, 627, 1817, fw=1024, fh=768)))
    assert any(isinstance(a, TapKeyIntent) and a.key == "esc" and a.reason == "ensure_mouse_mode_esc_1" for a in a_m1)
    fsm.step(
        tr.update(
            PerceptionSnapshot(
                timestamp=31.40, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
                coords=CoordReading(x=627, y=1817, confidence=0.99), minimap=MinimapReading(),
                interface_mode="mouse_banner",
            )
        )
    )
    assert fsm.state == BotState.SEARCH_MOB
