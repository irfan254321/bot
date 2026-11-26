"""Unit tests for Mouse-Mode 1-click combat, Auto-ESC guard, Anti-Miss approach guard, 3-retry loot cap & AHK v2 bridge."""

from __future__ import annotations

from vikings_bot.brain.fsm import FarmingFSM
from vikings_bot.capture.window import GameWindow
from vikings_bot.cli import build_parser
from vikings_bot.config import load_config
from vikings_bot.input.actions import ClickScreenIntent, DragCameraIntent, TapKeyIntent, TypeChatCommandIntent
from vikings_bot.input.ahk_driver import AhkInputDriver
from vikings_bot.state.models import (
    BotMode,
    BotState,
    CoordReading,
    LootDrop,
    MinimapReading,
    MobTarget,
    PerceptionSnapshot,
)
from vikings_bot.state.tracker import WorldStateTracker


def _mob(
    cx: int = 960,
    cy: int = 440,
    dist: float = 180.0,
    has_hp: bool = False,
    hp: float | None = None,
    contested: bool = False,
    contest_reason: str = "",
) -> MobTarget:
    return MobTarget(
        nameplate_bbox=(cx - 38, cy - 55, 76, 12),
        body_bbox=(cx - 45, cy - 38, cx + 45, cy + 72),
        click_x=cx,
        click_y=cy,
        foot_x=cx,
        foot_y=cy + 72,
        dist_to_player_px=dist,
        has_hp_bar=has_hp,
        hp_ratio_est=hp,
        contested=contested,
        contest_reason=contest_reason,
    )


def _bag(cx: int = 965, cy: int = 565, dist: float = 45.0) -> LootDrop:
    return LootDrop(bbox=(cx - 8, cy - 8, 16, 16), click_x=cx, click_y=cy, dist_to_player_px=dist, score=0.9)


def _snap(
    ts: float,
    x: int = 615,
    y: int = 1830,
    *,
    fw: int = 1920,
    fh: int = 1080,
    hp: float = 1.0,
    mobs: tuple[MobTarget, ...] = (),
    contested: tuple[MobTarget, ...] = (),
    loot: tuple[LootDrop, ...] = (),
    interface_mode: str = "mouse_cursor",
    pickup_changed: bool = False,
    respawn_dialog_bbox: tuple[int, int, int, int] | None = None,
    playfield_luma: float | None = None,
    sky_ratio: float | None = None,
    horizon_y: float | None = None,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        timestamp=ts,
        frame_width=fw,
        frame_height=fh,
        hp_ratio=hp,
        mp_ratio=1.0,
        coords=CoordReading(x=x, y=y, confidence=0.98),
        minimap=MinimapReading(),
        mobs=mobs,
        contested_mobs=contested,
        loot=loot,
        respawn_dialog_bbox=respawn_dialog_bbox,
        interface_mode=interface_mode,
        pickup_log_changed=pickup_changed,
        playfield_luma=playfield_luma,
        sky_ratio=sky_ratio,
        horizon_y=horizon_y,
    )


def _load_legacy_mouse_loot_cfg():
    """Helper for legacy tests that specifically exercise optional OpenCV GoldBag mouse-click loot mode."""
    c = load_config("config/default.yaml")
    c.perception["yolo_enabled"] = False
    c.farming["background_e_spam"] = False
    c.farming["disable_gold_bag_detector"] = False
    c.farming["engage_double_click"] = False
    c.farming["engage_skill_key"] = ""
    c.farming["return_center_min_drift_units"] = 0.0
    c.farming["post_kill_loot_window_sec"] = 10.0
    c.farming["loot_min_window_sec"] = 2.0
    c.farming["loot_empty_exit_sec"] = 1.2
    return c


def test_mouse_mode_single_click_engage_zero_spam_until_kill_then_loot_and_mmb_sweep():
    cfg = _load_legacy_mouse_loot_cfg()
    assert cfg.controls["mode"] == "mouse"
    assert cfg.controls["input_backend"] == "ahk"
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    assert fsm.state == BotState.SEARCH_MOB

    # 1. Gold Rustler spotted -> clicks LEFT exactly ONCE on upper-torso and enters COMBAT_WAIT!
    m1 = _mob(cx=1040, cy=450, dist=240.0, has_hp=False)
    acts_eng = fsm.step(tracker.update(_snap(1.5, mobs=(m1,))))
    assert fsm.state == BotState.COMBAT_WAIT
    assert len(acts_eng) == 1
    assert isinstance(acts_eng[0], ClickScreenIntent)
    assert acts_eng[0].button == "left" and acts_eng[0].x == 1040 and acts_eng[0].y == 450 and acts_eng[0].no_jitter

    # 2. During COMBAT_WAIT (HP dropping 0.90 -> 0.55 -> 0.15) -> ZERO extra clicks or keys!
    for ts, hp_v in ((2.0, 0.90), (2.6, 0.55), (3.2, 0.15)):
        m_hit = _mob(cx=1020, cy=460, dist=95.0, has_hp=True, hp=hp_v)
        distractor = _mob(cx=600, cy=420, dist=310.0, has_hp=False)
        acts_wait = fsm.step(tracker.update(_snap(ts, mobs=(distractor, m_hit))))
        assert fsm.state == BotState.COMBAT_WAIT
        assert acts_wait == [], f"Must NOT click again while mob is alive: {acts_wait}"
        assert tracker.target_lock.locked_mob is not None and tracker.target_lock.locked_mob.click_x == 1020

    # 3. Monster dies (HP gone, 2 Gold Bags drop near kill pos) -> enters LOOT & records last_kill_pos!
    far_bag = _bag(cx=700, cy=520, dist=210.0)
    kill_bag = _bag(cx=1010, cy=470, dist=140.0)
    fsm.step(tracker.update(_snap(3.7, mobs=(), loot=(far_bag, kill_bag))))
    assert fsm.state == BotState.LOOT
    assert fsm.kills_count == 1
    assert tracker.last_kill_pos == (1020, 460)

    # 4. Inside 800ms window, drop within 140px of last_kill_pos is prioritized first -> clicks bag once!
    w_l1 = tracker.update(_snap(3.9, mobs=(), loot=(far_bag, kill_bag)))
    assert w_l1.loot[0].click_x == 1010 and w_l1.loot[0].near_last_kill is True
    acts_l1 = fsm.step(w_l1)
    assert any(isinstance(a, ClickScreenIntent) and a.x == 1010 and a.y == 470 for a in acts_l1)

    # 5. Once Gold Bag is close to feet (<= 85px) -> taps 'e' cleanly!
    close_bag = _bag(cx=962, cy=560, dist=42.0)
    acts_l2 = fsm.step(tracker.update(_snap(4.5, mobs=(), loot=(close_bag,))))
    assert any(isinstance(a, TapKeyIntent) and a.key == "e" for a in acts_l2)

    # 6. Bags cleared (min 2.0s + 1.2s consecutive empty in LOOT) -> RETURN_TO_CENTER -> arrives (594,1832) -> SEARCH_MOB -> MMB sweep!
    fsm.step(tracker.update(_snap(5.0, mobs=(), loot=())))
    assert fsm.state == BotState.LOOT  # elapsed 1.3s < 2.0s
    fsm.step(tracker.update(_snap(6.0, mobs=(), loot=())))
    assert fsm.state == BotState.LOOT  # empty 1.0s < 1.2s
    fsm.step(tracker.update(_snap(6.6, mobs=(), loot=())))
    assert fsm.state == BotState.RETURN_TO_CENTER
    fsm.step(tracker.update(_snap(6.8, x=594, y=1832, mobs=(), loot=())))
    assert fsm.state == BotState.SEARCH_MOB

    acts_sweep = fsm.step(tracker.update(_snap(7.0, x=594, y=1832, mobs=(), loot=())))
    assert any(isinstance(a, DragCameraIntent) and a.button == "middle" and a.dy == 0 for a in acts_sweep)


def test_auto_esc_guard_switches_keyboard_banner_back_to_mouse_mode():
    """Verify one-shot startup ENSURE_MOUSE_MODE (1x or 2x ESC) and ZERO in-loop ESC during farming."""
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.ui_matcher import UiTemplateMatcher

    cfg = load_config("config/default.yaml")

    # 1. Width-based top-center INTERFACE TYPE banner detection at x=395..585, y=91..109 (1024x768)
    #    Uses real ref_mode_mouse.jpg / ref_mode_keyboard.jpg if present, else calibrated 1024x768 synthetic banner.
    from pathlib import Path
    import cv2

    cfg_vm = load_config("config/default.yaml")
    cfg_vm.perception["hud_fixed_pixel_ui"] = True
    um = UiTemplateMatcher(cfg_vm)

    fix_dir = Path(__file__).parent / "fixtures"
    ref_mouse = fix_dir / "ref_mode_mouse.jpg"
    ref_kb = fix_dir / "ref_mode_keyboard.jpg"
    if ref_mouse.exists():
        img_ms = cv2.imread(str(ref_mouse), cv2.IMREAD_COLOR)
    else:
        img_ms = np.full((768, 1024, 3), (25, 25, 25), dtype=np.uint8)
        img_ms[96:105, 455:529] = (235, 235, 235)  # 74px in [55..92px] -> Mouse
    mode_ms, _ = um.detect_interface_mode(FramePacket(image_bgr=img_ms, timestamp=1.0))
    assert mode_ms == "mouse_banner"

    if ref_kb.exists():
        img_kb = cv2.imread(str(ref_kb), cv2.IMREAD_COLOR)
    else:
        img_kb = np.full((768, 1024, 3), (25, 25, 25), dtype=np.uint8)
        img_kb[96:105, 430:550] = (235, 235, 235)  # 120px in [98..150px] -> Keyboard
    mode_kb, _ = um.detect_interface_mode(FramePacket(image_bgr=img_kb, timestamp=1.0))
    assert mode_kb == "keyboard_banner"

    # 2. Branch A: resume() -> ENSURE_MOUSE_MODE -> 1st ESC -> 0.32s -> mouse_banner -> SEARCH_MOB (no 2nd ESC!)
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0)
    assert fsm.state == BotState.ENSURE_MOUSE_MODE

    acts_1 = fsm.step(tracker.update(_snap(1.0, interface_mode="no_cursor")))
    assert len(acts_1) == 1 and isinstance(acts_1[0], TapKeyIntent) and acts_1[0].key == "esc"
    assert fsm.step(tracker.update(_snap(1.20, interface_mode="mouse_banner"))) == []  # waiting 0.32s
    assert fsm.state == BotState.ENSURE_MOUSE_MODE
    assert fsm.step(tracker.update(_snap(1.35, interface_mode="mouse_banner"))) == []
    assert fsm.state == BotState.SEARCH_MOB
    assert fsm._mouse_mode_verified is True

    # Once in SEARCH_MOB, even if keyboard_banner appears, NO in-loop ESC ping-pong!
    m = _mob(cx=980, cy=450, dist=160.0)
    acts_farm = fsm.step(tracker.update(_snap(1.6, mobs=(m,), interface_mode="keyboard_banner")))
    assert not any(isinstance(a, TapKeyIntent) and a.key == "esc" for a in acts_farm)
    assert fsm.state == BotState.COMBAT_WAIT

    # 3. Branch B: 1st ESC -> 0.32s -> keyboard_banner -> 2nd ESC -> 0.28s -> SEARCH_MOB!
    tracker2 = WorldStateTracker()
    fsm2 = FarmingFSM(cfg, tracker2, mode=BotMode.FULL)
    fsm2.resume(10.0)
    acts_b1 = fsm2.step(tracker2.update(_snap(10.0, interface_mode="no_cursor")))
    assert len(acts_b1) == 1 and isinstance(acts_b1[0], TapKeyIntent) and acts_b1[0].key == "esc"
    acts_b2 = fsm2.step(tracker2.update(_snap(10.35, interface_mode="keyboard_banner")))
    assert len(acts_b2) == 1 and isinstance(acts_b2[0], TapKeyIntent) and acts_b2[0].key == "esc"
    assert fsm2.state == BotState.ENSURE_MOUSE_MODE
    assert fsm2.step(tracker2.update(_snap(10.50, interface_mode="mouse_banner"))) == []  # waiting 0.28s
    assert fsm2.step(tracker2.update(_snap(10.68, interface_mode="mouse_banner"))) == []
    assert fsm2.state == BotState.SEARCH_MOB
    assert fsm2._mouse_mode_verified is True


def test_watchdog_anti_miss_never_cancels_while_distance_is_actively_shrinking():
    """Refinement #1: Watchdog 2.4s MUST NOT cancel lock while dist_to_player_px keeps decreasing."""
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)

    # Click a far mob at dist=440px (coords stagnant/occluded, no HP bar yet)
    m0 = _mob(cx=1100, cy=400, dist=440.0, has_hp=False)
    fsm.step(tracker.update(_snap(1.0, x=615, y=1830, mobs=(m0,))))
    assert fsm.state == BotState.COMBAT_WAIT

    # Even past 2.4s (up to t=5.0s = 4.0s elapsed!), dist keeps shrinking every 0.8s -> MUST STAY LOCKED!
    for idx, (ts, dist_px) in enumerate(((1.8, 380.0), (2.6, 310.0), (3.4, 240.0), (4.2, 170.0), (5.0, 110.0))):
        m_app = _mob(cx=1090 - idx * 8, cy=410 + idx * 5, dist=dist_px, has_hp=False)
        acts = fsm.step(tracker.update(_snap(ts, x=615, y=1830, mobs=(m_app,))))
        assert fsm.state == BotState.COMBAT_WAIT, f"Canceled lock prematurely at t={ts} while dist shrank to {dist_px}!"
        assert acts == []

    # Now character truly stalls (neither dist shrinks nor coords move) for >= 2.4s -> Watchdog unlocks cleanly!
    m_stall = _mob(cx=1058, cy=430, dist=110.0, has_hp=False)
    fsm.step(tracker.update(_snap(6.5, x=615, y=1830, mobs=(m_stall,))))
    assert fsm.state == BotState.COMBAT_WAIT
    fsm.step(tracker.update(_snap(7.5, x=615, y=1830, mobs=(m_stall,))))
    assert fsm.state == BotState.SEARCH_MOB


def test_loot_max_3_attempts_blacklists_obstructed_bag_and_never_blocks_80m_repair():
    """Refinement #2: Obstructed Gold Bag fails max 3 tries -> blacklist_point(120s) -> transitions to REPAIR_RECALL!"""
    cfg = _load_legacy_mouse_loot_cfg()
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)

    # Simulate 80m repair timer already due while an obstructed bag is sitting on the ground after a kill
    fsm._active_farming_sec = 4810.0
    blocked_bag = _bag(cx=1120, cy=520, dist=165.0)

    # Enters LOOT strictly post-kill
    fsm.transition(BotState.LOOT, 1.5, "test_post_kill_loot")
    assert fsm.state == BotState.LOOT

    # Attempt 1, 2 (spaced past click cooldown 0.35s) -> on attempt #2 blacklists coord & exits LOOT!
    blacklisted = None
    for ts in (2.0, 2.6):
        fsm.step(tracker.update(_snap(ts, loot=(blocked_bag,))))
        if fsm.last_blacklisted_loot is not None:
            blacklisted = fsm.last_blacklisted_loot
    assert blacklisted == (1120.0, 520.0)
    assert fsm.state == BotState.SEARCH_MOB

    # Now bag is blacklisted (invisible in next snapshot) -> 80m REPAIR_RECALL triggers immediately!
    fsm.step(tracker.update(_snap(3.5, loot=())))
    assert fsm.state == BotState.REPAIR_RECALL


def test_ahk_input_driver_protocol_vm_foreground_vs_host_background_and_cli_flags():
    sent_cmds: list[str] = []

    def _fake_transport(cmd: str) -> str:
        sent_cmds.append(cmd)
        return "PONG" if cmd == "PING" else "OK"

    class _FakeWinTracker:
        def find_window(self) -> GameWindow:
            return GameWindow(hwnd=0x1234, pid=999, title="Vikings Online", left=100, top=50, width=1024, height=768)

    # 1. VM Foreground mode sends screen coordinates verbatim
    drv_vm = AhkInputDriver(
        ahk_mode="vm_foreground",
        window_tracker=_FakeWinTracker(),  # type: ignore[arg-type]
        command_transport=_fake_transport,
    )
    ok, rep = drv_vm.health_check()
    assert ok is True and rep == "PONG"
    drv_vm.mouse_click(512, 384, button="left", hold_sec=0.04)
    drv_vm.tap_key("e", hold_sec=0.04)
    drv_vm.mouse_drag_rel(dx=185, dy=0, button="middle", steps=10, start_x=612, start_y=434)
    drv_vm.send_chat_command("/kill")
    drv_vm.release_all()

    assert "CLICK|vm_foreground|4660|512|384|left|40" in sent_cmds
    assert "KEY_TAP|vm_foreground|4660|e|40" in sent_cmds
    assert "DRAG_MMB|vm_foreground|4660|612|434|185|0|10" in sent_cmds
    assert "CHAT_CMD|vm_foreground|4660|/kill" in sent_cmds
    assert "RELEASE_ALL|4660" in sent_cmds

    # 2. Host Background mode converts screen (512, 384) -> client-relative (412, 334) using window origin (100, 50)
    sent_cmds.clear()
    drv_bg = AhkInputDriver(
        ahk_mode="host_background",
        window_tracker=_FakeWinTracker(),  # type: ignore[arg-type]
        command_transport=_fake_transport,
    )
    drv_bg.mouse_click(512, 384, button="left", hold_sec=0.04)
    drv_bg.tap_key("esc", hold_sec=0.05)
    assert sent_cmds[0] == "CLICK|host_background|4660|412|334|left|40"
    assert sent_cmds[1] == "KEY_TAP|host_background|4660|esc|50"

    # 3. CLI parses --config before/after subcommand, --target, --vm-1024, --bg-mode, and test-ahk-bg cleanly
    parser = build_parser()
    ns_run = parser.parse_args(
        ["run", "--config", "config/default.yaml", "--target", "gold", "--live", "--vm-1024", "--debug-overlay"]
    )
    assert (
        ns_run.config == "config/default.yaml"
        and ns_run.target == "gold"
        and ns_run.live is True
        and ns_run.vm_1024 is True
        and ns_run.debug_overlay is True
        and ns_run.bg_mode is False
    )
    ns_bg = parser.parse_args(["test-ahk-bg", "--config", "config/default.yaml", "--send-probe"])
    assert ns_bg.command == "test-ahk-bg" and ns_bg.config == "config/default.yaml" and ns_bg.send_probe is True
    ns_pre = parser.parse_args(["--config", "custom.yaml", "doctor", "--vm-1024"])
    assert ns_pre.command == "doctor" and ns_pre.config == "custom.yaml" and ns_pre.vm_1024 is True

    # 4. 64-bit-safe PrintWindow / GetDIBits smoke test on desktop HWND (no ctypes OverflowError!)
    import ctypes
    from vikings_bot.capture.screen import PrintWindowCaptureSource

    desktop_hwnd = int(ctypes.windll.user32.GetDesktopWindow())
    bgr = PrintWindowCaptureSource.grab_hwnd_bgr(desktop_hwnd, 128, 96)
    assert bgr is None or bgr.shape == (96, 128, 3)


def test_anti_false_death_guard_requires_5_frames_and_low_hp_or_respawn_or_town_warp():
    """Verify Anti-False-Death guard blocks glitchy 0% HP while allowing real confirmed deaths."""
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker(death_zero_hp_frames=5)
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)

    # 1. Alive at farming spot (610, 1841) with high HP (5099/5700 = 89.5%)
    fsm.step(tracker.update(_snap(1.0, x=610, y=1841, fw=1024, fh=768, hp=0.895)))
    assert fsm.state == BotState.SEARCH_MOB

    # 2. Glitch frames: 1..10 consecutive 0% HP straight from 89.5% with NO respawn dialog & NO town warp -> BLOCKED!
    for i in range(10):
        w = tracker.update(_snap(1.1 + i * 0.05, x=610, y=1841, fw=1024, fh=768, hp=0.0))
        fsm.step(w)
        assert w.is_dead is False
        assert fsm.state == BotState.SEARCH_MOB
        assert fsm.deaths_count == 0

    # 3. Recover high HP, then REAL death via prior HP drop < 35% (28%) -> requires >= 5 consecutive 0% frames
    fsm.step(tracker.update(_snap(2.0, x=610, y=1841, fw=1024, fh=768, hp=0.88)))
    fsm.step(tracker.update(_snap(2.2, x=610, y=1841, fw=1024, fh=768, hp=0.28)))
    for i in range(4):
        w = tracker.update(_snap(2.3 + i * 0.05, x=610, y=1841, fw=1024, fh=768, hp=0.0))
        fsm.step(w)
        assert fsm.state == BotState.SEARCH_MOB, f"Triggered death too early on frame {i + 1}/5"
    w5 = tracker.update(_snap(2.55, x=610, y=1841, fw=1024, fh=768, hp=0.0))
    fsm.step(w5)
    assert w5.is_dead is True
    assert fsm.state == BotState.DEAD_RESPAWN
    assert fsm.deaths_count == 1

    # 4. One-shot burst kill (HP 0.90 -> 0.0) corroborated by respawn dialog popup after 5 consecutive frames
    tracker2 = WorldStateTracker(death_zero_hp_frames=5)
    fsm2 = FarmingFSM(cfg, tracker2, mode=BotMode.FULL)
    fsm2.resume(10.0, ensure_mouse_mode=False)
    fsm2.step(tracker2.update(_snap(10.0, x=610, y=1841, fw=1024, fh=768, hp=0.90)))
    for i in range(4):
        fsm2.step(
            tracker2.update(
                _snap(10.1 + i * 0.05, x=610, y=1841, fw=1024, fh=768, hp=0.0, respawn_dialog_bbox=(480, 410, 55, 25))
            )
        )
        assert fsm2.state == BotState.SEARCH_MOB
    fsm2.step(
        tracker2.update(_snap(10.35, x=610, y=1841, fw=1024, fh=768, hp=0.0, respawn_dialog_bbox=(480, 410, 55, 25)))
    )
    assert fsm2.state == BotState.DEAD_RESPAWN
    assert fsm2.deaths_count == 1

    # 5. One-shot kill corroborated by actual coordinate warp to town (1075, 2628) after 5 consecutive 0% frames
    tracker3 = WorldStateTracker(death_zero_hp_frames=5)
    fsm3 = FarmingFSM(cfg, tracker3, mode=BotMode.FULL)
    fsm3.resume(20.0, ensure_mouse_mode=False)
    fsm3.step(tracker3.update(_snap(20.0, x=610, y=1841, fw=1024, fh=768, hp=0.90)))
    for i in range(5):
        # 3+ consistent town frames accepted as real teleport jump by WorldStateTracker
        fsm3.step(tracker3.update(_snap(20.2 + i * 0.05, x=1075, y=2628, fw=1024, fh=768, hp=0.0)))
    assert fsm3.state == BotState.DEAD_RESPAWN
    assert fsm3.deaths_count == 1


def test_loot_entry_gate_blocks_far_drops_without_kill_and_24px_grid_caps_jittery_bag():
    """Verify SEARCH_MOB never enters LOOT directly (post-kill only), and 24x24px grid blacklists jittery bags on try #3."""
    cfg = _load_legacy_mouse_loot_cfg()
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)

    # 1. Even with Gold Bags on the ground in SEARCH_MOB (no kill from COMBAT_WAIT), MUST stay in SEARCH_MOB!
    far_drop = _bag(cx=820, cy=480, dist=290.0)
    close_drop = _bag(cx=510, cy=500, dist=140.0)
    fsm.step(tracker.update(_snap(1.5, fw=1024, fh=768, loot=(far_drop, close_drop))))
    assert fsm.state == BotState.SEARCH_MOB

    # 2. Inside post-kill LOOT, a bag that jitters by +-4px across 3 attempts is quantized to 24x24px grid -> blacklist!
    tracker2 = WorldStateTracker()
    fsm2 = FarmingFSM(cfg, tracker2, mode=BotMode.FULL)
    fsm2.resume(10.0, ensure_mouse_mode=False)
    fsm2.transition(BotState.LOOT, 10.5, "test_post_kill_loot")
    jitter_bags = [
        _bag(cx=510, cy=500, dist=180.0),
        _bag(cx=516, cy=505, dist=182.0),
        _bag(cx=513, cy=503, dist=181.0),
    ]
    assert fsm2.state == BotState.LOOT

    blacklisted = None
    for idx, ts in enumerate((11.0, 11.6, 12.2)):
        fsm2.step(tracker2.update(_snap(ts, fw=1024, fh=768, loot=(jitter_bags[idx],))))
        if fsm2.last_blacklisted_loot is not None:
            blacklisted = fsm2.last_blacklisted_loot
    assert blacklisted is not None
    assert fsm2.state == BotState.RETURN_TO_CENTER


def test_auto_daylight_weather_sends_setweather_in_scan_spot_and_never_in_combat_wait():
    """Verify `/setweather 11` fires on resume/dark luma/600s interval ONLY in SEARCH_MOB, never during COMBAT_WAIT."""
    from pathlib import Path
    import cv2
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = load_config("config/default.yaml")
    assert cfg.farming["auto_daylight_weather"] is True
    assert cfg.farming["daylight_weather_cmd"] == "/setweather 11"
    assert float(cfg.farming["daylight_weather_interval_sec"]) == 600.0
    assert float(cfg.farming["daylight_dark_luma_threshold"]) == 68.0

    # Real night fixture vm_1024_raw.jpg must produce playfield_luma < 68.0
    vm_fix = Path(__file__).parent / "fixtures" / "vm_1024_raw.jpg"
    if vm_fix.exists():
        cfg_vm = load_config("config/default.yaml")
        cfg_vm.perception["hud_fixed_pixel_ui"] = True
        snap_vm = PerceptionPipeline(cfg_vm).analyze(
            FramePacket(image_bgr=cv2.imread(str(vm_fix)), timestamp=1.0, source_name="vm_1024_raw.jpg")
        )
        assert snap_vm.playfield_luma is not None and snap_vm.playfield_luma < 68.0

    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(10.0, ensure_mouse_mode=False)

    # 1. First entry into SEARCH_MOB after resume (F9) -> sends `/setweather 11`
    acts_init = fsm.step(tracker.update(_snap(10.5, fw=1024, fh=768, playfield_luma=92.0)))
    assert any(isinstance(a, TypeChatCommandIntent) and a.command == "/setweather 11" for a in acts_init)

    # 2. Immediately after (10s later) even if still dark (<68.0) -> respects 45s cooldown, does NOT spam!
    acts_cd = fsm.step(tracker.update(_snap(20.5, fw=1024, fh=768, playfield_luma=45.0)))
    assert not any(isinstance(a, TypeChatCommandIntent) for a in acts_cd)

    # 3. Engage a Gold Rustler -> enters COMBAT_WAIT; even when dark AND >=600s elapsed, NEVER sends `/setweather 11`!
    m = _mob(cx=510, cy=320, dist=140.0, has_hp=False)
    acts_eng = fsm.step(tracker.update(_snap(25.0, fw=1024, fh=768, mobs=(m,), playfield_luma=95.0)))
    assert fsm.state == BotState.COMBAT_WAIT
    assert any(isinstance(a, ClickScreenIntent) for a in acts_eng)

    m_alive = _mob(cx=510, cy=320, dist=95.0, has_hp=True, hp=0.65)
    acts_combat = fsm.step(tracker.update(_snap(640.0, fw=1024, fh=768, mobs=(m_alive,), playfield_luma=38.0)))
    assert not any(isinstance(a, TypeChatCommandIntent) for a in acts_combat)

    # 4. Return to SEARCH_MOB: dark playfield_luma (< 68.0) past 45s cooldown -> sends `/setweather 11`!
    fsm.transition(BotState.SEARCH_MOB, 70.0, "test_back_to_scan")
    acts_dark = fsm.step(tracker.update(_snap(70.5, fw=1024, fh=768, playfield_luma=44.0)))
    assert any(
        isinstance(a, TypeChatCommandIntent) and a.command == "/setweather 11" and "dark" in a.reason
        for a in acts_dark
    )

    # 5. Bright daylight (luma=96.0) but 600s timer elapsed -> sends periodic `/setweather 11`!
    acts_int = fsm.step(tracker.update(_snap(671.0, fw=1024, fh=768, playfield_luma=96.0)))
    assert any(
        isinstance(a, TypeChatCommandIntent) and a.command == "/setweather 11" and "interval" in a.reason
        for a in acts_int
    )


def test_ahk_pipe_errno22_fallback_f9_stuck_reset_and_nav_mob_priority():
    """Verify AHK Errno 22 falls back to Win32, F9 resets consecutive_stuck, and NAVIGATE_TO_SPOT engages mobs <=60u."""
    from vikings_bot.input.driver import DryRunInputDriver
    from vikings_bot.runtime.safety import SafetyMonitor

    # 1. AHK pipe throws ERR|[Errno 22] Invalid argument -> Win32 fallback executes click, ESC, MMB drag, and chat cmd!
    fb = DryRunInputDriver()
    drv = AhkInputDriver(
        ahk_mode="vm_foreground",
        command_transport=lambda _cmd: "ERR|[Errno 22] Invalid argument",
        fallback_driver=fb,
    )
    drv.mouse_click(512, 340, button="left", hold_sec=0.04)
    drv.tap_key("esc", hold_sec=0.05)
    drv.mouse_drag_rel(dx=160, dy=0, button="middle", steps=8, start_x=512, start_y=384)
    drv.send_chat_command("/setweather 11")
    assert any("mouse_click" in h and "512" in h and "340" in h for h in fb.history), fb.history
    assert any("tap_key" in h and "esc" in h for h in fb.history), fb.history
    assert any("mouse_drag_rel" in h and "160" in h for h in fb.history), fb.history
    assert any("send_chat_command(/setweather 11)" in h for h in fb.history), fb.history

    # 2. F9 resume() resets consecutive_stuck=0 & stuck timer so SafetyMonitor never auto-pauses on next tick!
    cfg = load_config("config/default.yaml")
    assert float(cfg.farming["stuck_timeout_sec"]) >= 3.5
    tracker = WorldStateTracker(stuck_timeout_sec=float(cfg.farming["stuck_timeout_sec"]))
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    safety = SafetyMonitor(cfg, enable_hotkeys=False)

    tracker._consecutive_stuck = 6
    tracker._is_stuck = True
    fsm.resume(5.0, ensure_mouse_mode=False)
    w_resumed = tracker.update(_snap(5.1, x=615, y=1830, fw=1024, fh=768))
    assert w_resumed.consecutive_stuck == 0
    assert w_resumed.is_stuck is False
    chk = safety.check(w_resumed, is_running_active=True)
    assert chk.force_pause_reason is None

    # 3. NAVIGATE_TO_SPOT within <= 60 coord units of spot with visible mob -> engages directly (COMBAT_WAIT)!
    spot = fsm.navigator.spot_center
    fsm.transition(BotState.NAVIGATE_TO_SPOT, 6.0, "test_nav_near_spot")
    m_near = _mob(cx=520, cy=330, dist=160.0, has_hp=False)
    acts_nav = fsm.step(
        tracker.update(_snap(6.2, x=int(spot.x + 40), y=int(spot.y + 15), fw=1024, fh=768, mobs=(m_near,)))
    )
    assert fsm.state == BotState.COMBAT_WAIT
    assert any(isinstance(a, ClickScreenIntent) and a.x == 520 and a.y == 330 for a in acts_nav)


def test_combat_wait_disables_mob_detection_and_loot_respects_220px_radius():
    """Verify skip_mob_detection=True during COMBAT_WAIT (no mob switching) & <=220px loot radius cap."""
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg_vm = load_config("config/default.yaml")
    cfg_vm.perception["hud_fixed_pixel_ui"] = True
    cfg_vm.perception["yolo_enabled"] = False
    pipe_vm = PerceptionPipeline(cfg_vm)

    # 1. Frame with a Gold Rustler nameplate + green target HP bar + 2 Gold Bags (1 close <=220px, 1 far >220px)
    hsv_bg = np.zeros((768, 1024, 3), dtype=np.uint8)
    hsv_bg[:, :] = (28, 120, 60)
    bgr = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    # Nameplate + green HP bar near center
    np_patch = np.full((10, 58, 3), (160, 110, 210), dtype=np.uint8)
    np_patch[:, ::2] = (28, 120, 50)
    bgr[300:310, 485:543] = cv2.cvtColor(np_patch, cv2.COLOR_HSV2BGR)
    bgr[314:318, 490:542] = cv2.cvtColor(np.full((4, 52, 3), (62, 220, 225), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    # Close Gold Bag at (420, 440) (~95px from feet (512, 424))
    bgr[444:456, 416:426] = cv2.cvtColor(np.full((12, 10, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr[440:444, 419:424] = cv2.cvtColor(np.full((4, 5, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    # Far Gold Bag at (160, 440) (~352px from feet -> must be ignored when loot_radius_limit_px=220.0!)
    bgr[444:456, 156:166] = cv2.cvtColor(np.full((12, 10, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr[440:444, 159:164] = cv2.cvtColor(np.full((4, 5, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)

    snap_combat = pipe_vm.analyze(
        FramePacket(image_bgr=bgr, timestamp=1.0),
        skip_mob_detection=True,
        loot_radius_limit_px=220.0,
        check_interface_mode=False,
    )
    assert snap_combat.mobs == (), "GoldRustlerDetector must be disabled during COMBAT_WAIT!"
    assert snap_combat.combat_target_hp_visible is True
    assert snap_combat.target_hp_visible is True
    assert len(snap_combat.loot) == 1 and abs(snap_combat.loot[0].click_x - 421) <= 12

    # 2. FSM in COMBAT_WAIT with skip_mob_detection=True (mobs=()): stays locked while HP visible, then enters LOOT on kill!
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg_vm, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    m_close = _mob(cx=515, cy=340, dist=90.0)
    m_far = _mob(cx=260, cy=320, dist=280.0)
    # Picks nearest mob from (512, 384) regardless of input list order!
    acts_e = fsm.step(tracker.update(_snap(1.2, fw=1024, fh=768, mobs=(m_far, m_close))))
    assert fsm.state == BotState.COMBAT_WAIT
    assert isinstance(acts_e[0], ClickScreenIntent) and acts_e[0].x == 515

    # HP visible via combat_target_hp_visible while mobs=() (first hit brings HP to 92%)
    snap_hp = PerceptionSnapshot(
        timestamp=2.0, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
        coords=CoordReading(x=615, y=1830, confidence=0.98), minimap=MinimapReading(),
        mobs=(), combat_target_hp_visible=True, combat_target_hp_ratio=0.92,
    )
    assert fsm.step(tracker.update(snap_hp)) == []
    assert fsm.state == BotState.COMBAT_WAIT

    # HP disappears -> after combat_zero_drop_kill_sec transitions directly to LOOT!
    snap_dead = PerceptionSnapshot(
        timestamp=3.8, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
        coords=CoordReading(x=615, y=1830, confidence=0.98), minimap=MinimapReading(),
        mobs=(), combat_target_hp_visible=False, combat_target_hp_ratio=None,
    )
    fsm.step(tracker.update(snap_dead))
    assert fsm.state == BotState.LOOT
    assert fsm.kills_count == 1

    # 3. Overlay hides cyan LOCK box during COMBAT_WAIT, AND Anti-Ghost-Lock aborts after 1.5s -> 15s blacklist!
    from vikings_bot.debug_overlay import build_overlay_frame_data

    tracker3 = WorldStateTracker()
    fsm3 = FarmingFSM(cfg_vm, tracker3, mode=BotMode.FULL)
    fsm3.resume(20.0, ensure_mouse_mode=False)
    ghost = _mob(cx=380, cy=310, dist=170.0)
    fsm3.step(tracker3.update(_snap(20.1, x=615, y=1830, fw=1024, fh=768, mobs=(ghost,))))
    assert fsm3.state == BotState.COMBAT_WAIT
    w_cw = tracker3.update(_snap(20.5, x=615, y=1830, fw=1024, fh=768, mobs=()))
    ov_cw = build_overlay_frame_data(
        game_rect=(0, 0, 1024, 768), world=w_cw, fsm_state=BotState.COMBAT_WAIT, config=cfg_vm, loop_hz=20.0
    )
    assert ov_cw.locked_mob is None and ov_cw.mobs == (), "Overlay must hide LOCK box during COMBAT_WAIT!"

    # After >=2.0s min lock hold with no movement & no THP -> Anti-Ghost-Lock cancels COMBAT_WAIT and blacklists (380, 310) for 15s!
    fsm3.step(tracker3.update(_snap(22.2, x=615, y=1830, fw=1024, fh=768, mobs=())))
    assert fsm3.state == BotState.SEARCH_MOB
    acts_skip = fsm3.step(tracker3.update(_snap(25.0, x=615, y=1830, fw=1024, fh=768, mobs=(ghost,))))
    assert fsm3.state == BotState.SEARCH_MOB and not any(isinstance(a, ClickScreenIntent) for a in acts_skip)


def test_auto_camera_pitch_leveling_and_12s_anti_oscillation_lockout():
    """Verify vertical pitch DRAG_MMB (dx=0, dy=+-60), safe zone no-op (horizon_y=190), and 12s opposite lockout."""
    cfg = load_config("config/default.yaml")
    assert cfg.farming["auto_camera_pitch"] is True
    assert float(cfg.farming["pitch_horizon_min_y"]) == 105.0
    assert float(cfg.farming["pitch_horizon_max_y"]) == 300.0
    assert int(cfg.farming["pitch_step_dy"]) == 60
    assert float(cfg.farming["pitch_adjust_cooldown_sec"]) == 4.0
    assert float(cfg.farming["pitch_opposite_lockout_sec"]) == 12.0

    # (a) Camera pitched too far down (horizon_y=60 < 105) -> sends DRAG_MMB(dx=0, dy=-60)
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    acts_up = fsm.step(tracker.update(_snap(10.0, fw=1024, fh=768, horizon_y=60.0, sky_ratio=0.06)))
    assert len(acts_up) == 1
    assert isinstance(acts_up[0], DragCameraIntent) and acts_up[0].dx == 0 and acts_up[0].dy == -60

    # (d) Opposite direction (camera suddenly reads too high at t=15.0, only 5s after dy=-60 < 12.0s lockout) -> BLOCKED!
    acts_locked = fsm.step(tracker.update(_snap(15.0, fw=1024, fh=768, horizon_y=360.0, sky_ratio=0.50)))
    assert not any(isinstance(a, DragCameraIntent) and a.dx == 0 and a.dy == 60 for a in acts_locked)

    # (b) After 12.0s lockout expires (t=22.5 >= 10.0 + 12.0), camera pitched too high -> sends DRAG_MMB(dx=0, dy=+60)!
    acts_dn = fsm.step(tracker.update(_snap(22.5, fw=1024, fh=768, horizon_y=350.0, sky_ratio=0.48)))
    assert len(acts_dn) == 1
    assert isinstance(acts_dn[0], DragCameraIntent) and acts_dn[0].dx == 0 and acts_dn[0].dy == 60

    # (c) Safe zone (horizon_y=190, sky_ratio=0.25) -> sends NO vertical pitch input!
    tracker2 = WorldStateTracker()
    fsm2 = FarmingFSM(cfg, tracker2, mode=BotMode.FULL)
    fsm2.resume(1.0, ensure_mouse_mode=False)
    acts_ok, settling = fsm2.farm.maybe_build_camera_pitch_adjust(
        tracker2.update(_snap(10.0, fw=1024, fh=768, horizon_y=190.0, sky_ratio=0.25)),
        10.0,
    )
    assert acts_ok == [] and settling is False


def test_combat_wait_hard_locks_mouse_and_post_kill_10s_close_loot_window():
    """Verify (a) COMBAT_WAIT disables all object detectors + overlay boxes + mouse/camera output,
    and (b) post-kill LOOT runs up to 10.0s strictly for Gold Bags <=220px without locking other mobs."""
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.debug_overlay import build_overlay_frame_data
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = _load_legacy_mouse_loot_cfg()
    cfg.perception["hud_fixed_pixel_ui"] = True
    assert float(cfg.farming["post_kill_loot_window_sec"]) == 10.0
    assert int(cfg.farming["loot_close_radius_px"]) == 310

    # (a) Pipeline with skip_all_object_detectors=True (COMBAT_WAIT):
    #     disables GoldRustlerDetector, GoldBagDetector, and CorpseDetector, while reading Target HP!
    hsv_bg = np.zeros((768, 1024, 3), dtype=np.uint8)
    hsv_bg[:, :] = (28, 120, 60)
    bgr = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    # Nameplate + green target HP bar + close Gold Bag
    np_patch = np.full((10, 58, 3), (160, 110, 210), dtype=np.uint8)
    np_patch[:, ::2] = (28, 120, 50)
    bgr[300:310, 485:543] = cv2.cvtColor(np_patch, cv2.COLOR_HSV2BGR)
    bgr[314:318, 490:542] = cv2.cvtColor(np.full((4, 52, 3), (62, 220, 225), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr[444:456, 416:426] = cv2.cvtColor(np.full((12, 10, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr[440:444, 419:424] = cv2.cvtColor(np.full((4, 5, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)

    pipe = PerceptionPipeline(cfg)
    snap_cw = pipe.analyze(
        FramePacket(image_bgr=bgr, timestamp=1.0),
        skip_all_object_detectors=True,
        check_interface_mode=False,
    )
    assert snap_cw.mobs == ()
    assert snap_cw.loot == ()
    assert snap_cw.dead_corpses == ()
    assert snap_cw.combat_target_hp_visible is True
    assert snap_cw.target_hp_visible is True

    # Overlay in COMBAT_WAIT must hide LOCK, MOB, GOLD, and corpse boxes even if snapshot had data
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    m_eng = _mob(cx=510, cy=330, dist=95.0, has_hp=False)
    acts_eng = fsm.step(tracker.update(_snap(1.1, fw=1024, fh=768, mobs=(m_eng,))))
    assert fsm.state == BotState.COMBAT_WAIT
    assert len(acts_eng) == 1 and isinstance(acts_eng[0], ClickScreenIntent)

    w_cw = tracker.update(
        PerceptionSnapshot(
            timestamp=2.0,
            frame_width=1024,
            frame_height=768,
            hp_ratio=1.0,
            mp_ratio=1.0,
            coords=CoordReading(x=615, y=1830, confidence=0.98),
            minimap=MinimapReading(),
            mobs=(),
            loot=(),
            dead_corpses=(),
            combat_target_hp_visible=True,
            combat_target_hp_ratio=0.75,
            horizon_y=30.0,
            sky_ratio=0.02,
        )
    )
    ov_cw = build_overlay_frame_data(
        game_rect=(0, 0, 1024, 768), world=w_cw, fsm_state=BotState.COMBAT_WAIT, config=cfg, loop_hz=20.0
    )
    assert ov_cw.mobs == () and ov_cw.locked_mob is None and ov_cw.loot == () and ov_cw.corpses == ()

    # During COMBAT_WAIT: ZERO clicks, ZERO DRAG_MMB (even with extreme pitch!), ZERO ESC
    acts_cw = fsm.step(w_cw)
    assert acts_cw == []
    assert fsm.state == BotState.COMBAT_WAIT

    # (b) Target HP drops to 0% / disappears -> enters LOOT at t=3.8
    w_dead = tracker.update(
        PerceptionSnapshot(
            timestamp=3.8,
            frame_width=1024,
            frame_height=768,
            hp_ratio=1.0,
            mp_ratio=1.0,
            coords=CoordReading(x=615, y=1830, confidence=0.98),
            minimap=MinimapReading(),
            mobs=(),
            loot=(),
            combat_target_hp_visible=False,
            combat_target_hp_ratio=0.0,
        )
    )
    assert fsm.step(w_dead) == []
    assert fsm.state == BotState.LOOT
    assert fsm.kills_count == 1

    # During LOOT: another mob + far Gold Bag (350px > 310px) + close Gold Bag (130px <= 310px)
    other_mob = _mob(cx=320, cy=320, dist=200.0, has_hp=False)
    far_bag = _bag(cx=160, cy=420, dist=350.0)
    close_bag = _bag(cx=440, cy=450, dist=130.0)
    w_loot1 = tracker.update(_snap(4.2, fw=1024, fh=768, mobs=(other_mob,), loot=(far_bag, close_bag)))
    ov_loot = build_overlay_frame_data(
        game_rect=(0, 0, 1024, 768), world=w_loot1, fsm_state=BotState.LOOT, config=cfg, loop_hz=20.0
    )
    assert ov_loot.mobs == () and ov_loot.locked_mob is None, "Overlay must hide MOB/LOCK during LOOT!"
    acts_l1 = fsm.step(w_loot1)
    assert fsm.state == BotState.LOOT
    # Must click ONLY the close Gold Bag (440, 450), never the mob or far bag, and never DRAG_MMB!
    assert any(isinstance(a, ClickScreenIntent) and a.x == 440 and a.y == 450 for a in acts_l1)
    assert not any(isinstance(a, DragCameraIntent) for a in acts_l1)

    # Only far bag (>310px) remains -> ignored, stays in LOOT during initial 2.0s window (t=5.5 -> elapsed 1.7s)
    far_bag = _bag(cx=160, cy=420, dist=350.0)
    acts_l2 = fsm.step(tracker.update(_snap(5.5, fw=1024, fh=768, mobs=(other_mob,), loot=(far_bag,))))
    assert fsm.state == BotState.LOOT
    assert not any(isinstance(a, ClickScreenIntent) for a in acts_l2)

    # After >=2.0s elapsed AND >=1.2s consecutive empty (empty since 5.5 -> 7.1 is 1.6s empty) -> RETURN_TO_CENTER!
    fsm.step(tracker.update(_snap(7.1, fw=1024, fh=768, mobs=(other_mob,), loot=(far_bag,))))
    assert fsm.state == BotState.RETURN_TO_CENTER

    # Verify 10.0s hard post_kill_loot_window_sec cap when close bags keep appearing continuously
    fsm.transition(BotState.LOOT, 20.0, "test_10s_cap")
    for idx, ts in enumerate((21.0, 23.0, 25.0, 27.0, 29.5)):
        # Distinct close bag positions so 2-attempt blacklist never fires early
        b_i = _bag(cx=430 + idx * 35, cy=440, dist=120.0)
        fsm.step(tracker.update(_snap(ts, fw=1024, fh=768, loot=(b_i,))))
        assert fsm.state == BotState.LOOT
    fsm.step(tracker.update(_snap(30.1, fw=1024, fh=768, loot=(_bag(cx=520, cy=440, dist=110.0),))))
    assert fsm.state == BotState.RETURN_TO_CENTER


def test_combat_wait_hard_gate_post_kill_loot_e_tap_and_hybrid_green_hp_bar_detector():
    """Verify COMBAT_WAIT perception/FSM hard-gate, non-LOOT GoldBagDetector block, 0.45s E-tap,
    thin vertical armor rejection (+140px HP-bar guard), and 5-mob hybrid green HP-bar detection."""
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = _load_legacy_mouse_loot_cfg()
    cfg.perception["hud_fixed_pixel_ui"] = True
    pipe = PerceptionPipeline(cfg)

    # Build a 1024x768 frame with 5 living Gold Rustlers (green HP bars outside SELF) + 1 SELF bar + Gold Bags
    hsv_bg = np.zeros((768, 1024, 3), dtype=np.uint8)
    hsv_bg[:, :] = (28, 120, 60)
    bgr = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)

    green_bar = cv2.cvtColor(np.full((4, 52, 3), (62, 210, 210), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    mob_bars = [(180, 240), (310, 250), (660, 245), (790, 260), (240, 380)]
    for bx, by in mob_bars:
        bgr[by : by + 4, bx : bx + 52] = green_bar
    # Self bar inside SELF exclusion zone (x=430..595, y=305..480) -> MUST be ignored!
    bgr[360:364, 485:537] = green_bar
    # Close Gold Bag at (420, 440)
    bgr[444:454, 416:428] = cv2.cvtColor(np.full((10, 12, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr[440:444, 419:424] = cv2.cvtColor(np.full((4, 5, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)

    # 1. COMBAT_WAIT Hard-Gate in pipeline: NEVER invokes GoldRustlerDetector or GoldBagDetector!
    calls = {"mobs": 0, "loot": 0}
    orig_mobs = pipe.mobs.detect_with_context
    orig_loot = pipe.loot.detect

    def _spy_mobs(*args, **kwargs):
        calls["mobs"] += 1
        return orig_mobs(*args, **kwargs)

    def _spy_loot(*args, **kwargs):
        calls["loot"] += 1
        return orig_loot(*args, **kwargs)

    pipe.mobs.detect_with_context = _spy_mobs
    pipe.loot.detect = _spy_loot

    snap_cw = pipe.analyze(
        FramePacket(image_bgr=bgr, timestamp=1.0),
        current_state=BotState.COMBAT_WAIT,
        check_interface_mode=False,
    )
    assert len(snap_cw.mobs) == 0 and len(snap_cw.gold_bags) == 0 and len(snap_cw.dead_corpses) == 0
    assert calls["mobs"] == 0 and calls["loot"] == 0, f"Detectors must NOT be called in COMBAT_WAIT: {calls}"

    # 2. SEARCH_MOB: GoldRustlerDetector detects all 5 mobs via green HP bars (click_y = bar_y + 38),
    #    SELF bar ignored, AND GoldBagDetector is OFF (gold_bags == ())!
    snap_scan = pipe.analyze(
        FramePacket(image_bgr=bgr, timestamp=2.0),
        current_state=BotState.SEARCH_MOB,
        check_interface_mode=False,
    )
    assert len(snap_scan.mobs) == 5, f"Expected 5 hybrid green-HP-bar Gold Rustlers, got {len(snap_scan.mobs)}: {snap_scan.mobs}"
    assert len(snap_scan.gold_bags) == 0, "GoldBagDetector must be OFF outside BotState.LOOT!"
    assert calls["loot"] == 0
    for m in snap_scan.mobs:
        assert not (472 <= m.click_x <= 552 and 350 <= m.click_y <= 432)
        assert any(abs(m.click_x - (bx + 26)) <= 4 and abs(m.click_y - (by + 42)) <= 2 for bx, by in mob_bars)

    # 3. FSM COMBAT_WAIT Hard-Gate: injecting fake mobs + fake bags NEVER jumps to LOOT and returns len(actions)==0!
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    m_eng = _mob(cx=510, cy=330, dist=95.0, has_hp=False)
    fsm.step(tracker.update(_snap(1.1, fw=1024, fh=768, mobs=(m_eng,))))
    assert fsm.state == BotState.COMBAT_WAIT

    fake_mobs = (_mob(cx=200, cy=280, dist=240.0), _mob(cx=720, cy=280, dist=260.0))
    fake_bags = (_bag(cx=440, cy=450, dist=110.0), _bag(cx=540, cy=460, dist=95.0))
    w_fake = tracker.update(
        PerceptionSnapshot(
            timestamp=2.0,
            frame_width=1024,
            frame_height=768,
            hp_ratio=1.0,
            mp_ratio=1.0,
            coords=CoordReading(x=615, y=1830, confidence=0.98),
            minimap=MinimapReading(),
            mobs=fake_mobs,
            loot=fake_bags,
            combat_target_hp_visible=True,
            combat_target_hp_ratio=0.80,
        )
    )
    acts_cw = fsm.step(w_fake)
    assert fsm.state == BotState.COMBAT_WAIT
    assert len(acts_cw) == 0
    assert fsm.active_lock is None

    # 4. Post-kill LOOT taps E every 0.30s without DRAG_MMB
    fsm.transition(BotState.LOOT, 5.0, "test_e_tap")
    acts_e1 = fsm.step(tracker.update(_snap(5.0, fw=1024, fh=768, loot=())))
    assert any(isinstance(a, TapKeyIntent) and a.key == "e" for a in acts_e1)
    assert not any(isinstance(a, DragCameraIntent) for a in acts_e1)
    assert fsm.step(tracker.update(_snap(5.15, fw=1024, fh=768, loot=()))) == []  # < 0.30s cooldown
    acts_e2 = fsm.step(tracker.update(_snap(5.35, fw=1024, fh=768, loot=())))
    assert any(isinstance(a, TapKeyIntent) and a.key == "e" for a in acts_e2)

    # 5. GoldBagDetector rejects thin vertical monster-arm gleam (h > w * 1.25) AND gleam +130px below HP bar!
    bgr_arm = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    # Thin vertical golden arm gleam at (380, 420): 6px wide x 18px tall -> MUST BE REJECTED!
    bgr_arm[420:438, 380:386] = cv2.cvtColor(np.full((18, 6, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr_arm[438:450, 378:388] = cv2.cvtColor(np.full((12, 10, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    # Gleam +130px below a green HP bar at (600, 260) -> MUST BE MASKED by +140px HP bar guard!
    bgr_arm[260:264, 600:652] = green_bar
    bgr_arm[394:404, 620:632] = cv2.cvtColor(np.full((10, 12, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr_arm[390:394, 623:628] = cv2.cvtColor(np.full((4, 5, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    drops_arm, _ = orig_loot(FramePacket(image_bgr=bgr_arm, timestamp=6.0), mobs=(), max_dist_override_px=220.0)
    assert len(drops_arm) == 0, f"Vertical arm / +130px HP-bar armor gleam must be rejected: {drops_arm}"


def test_hit_confirmed_combat_mandatory_7s_loot_self_zone_and_nav_guard():
    """Verify Find -> Hit-Confirmed Combat (3s miss / 2.5s post-hit -> mandatory 7s LOOT) -> Find,
    expanded SELF zone (425..600, 300..555), and NAVIGATE_TO_SPOT <=25u mob guard."""
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = _load_legacy_mouse_loot_cfg()
    cfg.perception["hud_fixed_pixel_ui"] = True
    assert float(cfg.farming["combat_no_hit_abort_sec"]) == 5.0
    assert float(cfg.farming["combat_hit_confirm_timeout_sec"]) == 5.0
    assert float(cfg.farming["combat_confirmed_max_sec"]) == 35.0
    assert float(cfg.farming["combat_post_last_hit_idle_sec"]) == 2.5
    assert float(cfg.farming["post_kill_loot_window_sec"]) == 10.0

    pipe = PerceptionPipeline(cfg)

    # 1. Pipeline detects floating orange/yellow damage digits (height >= 16px) in x=390..635, y=200..360 during COMBAT_WAIT
    hsv_bg = np.zeros((768, 1024, 3), dtype=np.uint8)
    hsv_bg[:, :] = (28, 120, 60)
    bgr_hit = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    # Paint bright orange damage digits '128' (22px tall) at (480, 260)
    dmg_col = cv2.cvtColor(np.full((22, 12, 3), (20, 220, 240), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr_hit[260:282, 480:492] = dmg_col
    snap_hit = pipe.analyze(
        FramePacket(image_bgr=bgr_hit, timestamp=1.0),
        current_state=BotState.COMBAT_WAIT,
        check_interface_mode=False,
    )
    assert snap_hit.hit_damage_visible is True
    assert snap_hit.mobs == () and snap_hit.gold_bags == ()

    # Plain grass frame -> hit_damage_visible is False
    bgr_plain = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    snap_nohit = pipe.analyze(
        FramePacket(image_bgr=bgr_plain, timestamp=1.5),
        current_state=BotState.COMBAT_WAIT,
        check_interface_mode=False,
    )
    assert snap_nohit.hit_damage_visible is False

    # 2. Tight SELF zone (x=472..552, y=350..432): bar inside SELF ignored, legitimate mob outside SELF detected!
    bgr_self = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    green_bar = cv2.cvtColor(np.full((4, 52, 3), (62, 210, 210), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    # Bar inside tight SELF x=472..552, y=350..432
    bgr_self[368:372, 486:538] = green_bar
    # Legitimate Gold Rustler outside SELF at (260, 260)
    bgr_self[260:264, 234:286] = green_bar
    mobs_self = pipe.mobs.detect(FramePacket(image_bgr=bgr_self, timestamp=2.0))
    assert len(mobs_self) == 1 and abs(mobs_self[0].click_x - 260) <= 4

    # 3. Stage 1 Miss (no hit_damage_visible & no HP drop for 6.5s) -> blacklists 15s & returns to SEARCH_MOB!
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    grass_mob = _mob(cx=310, cy=320, dist=190.0, has_hp=False)
    fsm.step(tracker.update(_snap(1.0, x=615, y=1830, fw=1024, fh=768, mobs=(grass_mob,))))
    assert fsm.state == BotState.COMBAT_WAIT
    # Character walks briefly (coords change) but never sees hit_damage_visible or HP drop for 6.5s!
    fsm.step(tracker.update(_snap(4.5, x=617, y=1831, fw=1024, fh=768, mobs=())))
    assert fsm.state == BotState.COMBAT_WAIT
    fsm.step(tracker.update(_snap(7.6, x=618, y=1831, fw=1024, fh=768, mobs=())))
    assert fsm.state == BotState.SEARCH_MOB
    # Blacklisted for 15s: clicking same grass coord at t=12.0 is skipped!
    acts_bl = fsm.step(tracker.update(_snap(12.0, x=618, y=1831, fw=1024, fh=768, mobs=(grass_mob,))))
    assert fsm.state == BotState.SEARCH_MOB and not any(isinstance(a, ClickScreenIntent) for a in acts_bl)

    # 4. Stage 2 Hit-Confirmed Combat -> 2.5s post-last-hit idle -> MANDATORY 8.0s LOOT -> SEARCH_MOB!
    real_mob = _mob(cx=720, cy=310, dist=210.0, has_hp=False)
    fsm.step(tracker.update(_snap(20.0, x=615, y=1830, fw=1024, fh=768, mobs=(real_mob,))))
    assert fsm.state == BotState.COMBAT_WAIT
    # Hit damage numbers appear at t=21.0 and t=24.0 -> combat_confirmed=True, zero mouse/camera actions!
    for ts in (21.0, 22.5, 24.0):
        w_h = tracker.update(
            PerceptionSnapshot(
                timestamp=ts, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
                coords=CoordReading(x=615, y=1830, confidence=0.98), minimap=MinimapReading(),
                hit_damage_visible=True,
            )
        )
        assert fsm.step(w_h) == []
        assert fsm.state == BotState.COMBAT_WAIT and fsm._combat_confirmed is True
    # Still within 2.5s of last hit (t=26.0 -> 2.0s idle) -> stays in COMBAT_WAIT
    assert fsm.step(tracker.update(_snap(26.0, fw=1024, fh=768))) == []
    assert fsm.state == BotState.COMBAT_WAIT
    # At t=26.6 (2.6s >= 2.5s since last hit) -> MANDATORY transition to BotState.LOOT!
    fsm.step(tracker.update(_snap(26.6, fw=1024, fh=768)))
    assert fsm.state == BotState.LOOT
    # Post-kill LOOT: stays in LOOT during initial 2.0s window, exits after >=2.0s + 1.2s empty -> RETURN_TO_CENTER!
    acts_loot_e = fsm.step(tracker.update(_snap(27.0, fw=1024, fh=768)))
    assert fsm.state == BotState.LOOT
    assert any(isinstance(a, TapKeyIntent) and a.key == "e" for a in acts_loot_e)
    fsm.step(tracker.update(_snap(28.8, fw=1024, fh=768)))
    assert fsm.state == BotState.RETURN_TO_CENTER
    fsm.step(tracker.update(_snap(29.0, x=594, y=1832, fw=1024, fh=768)))
    assert fsm.state == BotState.SEARCH_MOB

    # 5. NAVIGATE_TO_SPOT guard: when spot_dist <= 25u and len(mobs) >= 1, stays in SEARCH_MOB & engages mob!
    from dataclasses import replace as dc_replace

    spot = fsm.navigator.spot_center
    # Force a tight farm_radius=10 so spot_dist=20u would normally trigger leash exit
    fsm.navigator.route = dc_replace(fsm.navigator.route, farm_radius=10.0)
    fsm.navigator.leash_exit_margin = 0.0
    m_near = _mob(cx=740, cy=320, dist=180.0, has_hp=False)
    acts_no_nav = fsm.step(
        tracker.update(_snap(35.0, x=int(spot.x + 20), y=int(spot.y), fw=1024, fh=768, mobs=(m_near,)))
    )
    assert fsm.state == BotState.COMBAT_WAIT
    assert any(isinstance(a, ClickScreenIntent) and a.x == 740 and a.y == 320 for a in acts_no_nav)


def test_combat_wait_survives_5s_chase_and_stagnant_hp_then_mandatory_8s_loot_and_spot_radius_55():
    """Verify watchdog_hp_stagnant never aborts during chase or after hit confirmation,
    spot farm_radius=55 blocks drift/patrol when mobs>=1 (<=85u), and AHK script has blocking stdin loop."""
    from pathlib import Path

    cfg = load_config("config/default.yaml")
    assert cfg.active_route.farm_radius == 55.0

    # 1. AHK v2 bridge script must NOT check gStdIn.AtEOF before ReadLine() (prevents instant empty_reply exit!)
    ahk_code = "\n".join(
        ln for ln in Path("ahk/vikings_input_bridge.ahk").read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith(";")
    )
    read_pos = ahk_code.find("gStdIn.ReadLine()")
    ateof_pos = ahk_code.find("gStdIn.AtEOF")
    assert read_pos > 0 and (ateof_pos == -1 or ateof_pos > read_pos), "gStdIn.AtEOF must only be checked AFTER ReadLine()!"

    # 2. During a 5.5s chase (pos moving 28 units) with full THP=1.0, watchdog_hp_stagnant MUST NOT abort at t=5.0s!
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    far_mob = _mob(cx=740, cy=310, dist=240.0, has_hp=False)
    fsm.step(tracker.update(_snap(1.0, x=615, y=1830, fw=1024, fh=768, mobs=(far_mob,))))
    assert fsm.state == BotState.COMBAT_WAIT

    for step_i in range(1, 6):
        ts = 1.0 + step_i * 0.9  # t=1.9, 2.8, 3.7, 4.6, 5.5 (first hit lands at t=4.6s < 5.0s abort!)
        w_chase = tracker.update(
            PerceptionSnapshot(
                timestamp=ts,
                frame_width=1024,
                frame_height=768,
                hp_ratio=1.0,
                mp_ratio=1.0,
                coords=CoordReading(x=615 + step_i * 5, y=1830, confidence=0.98),
                minimap=MinimapReading(),
                combat_target_hp_visible=True,
                combat_target_hp_ratio=1.0,
                local_hit_damage_visible=(step_i >= 4),  # first local hit lands at t=4.6s!
            )
        )
        assert fsm.step(w_chase) == []
        assert fsm.state == BotState.COMBAT_WAIT

    assert fsm._combat_confirmed is True

    # 3. Once hit_damage_visible confirmed combat, watchdog_hp_stagnant stays DISABLED even across 14s of combat!
    for ts in (8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5):
        w_fight = tracker.update(
            PerceptionSnapshot(
                timestamp=ts,
                frame_width=1024,
                frame_height=768,
                hp_ratio=1.0,
                mp_ratio=1.0,
                coords=CoordReading(x=645, y=1830, confidence=0.98),
                minimap=MinimapReading(),
                combat_target_hp_visible=True,
                combat_target_hp_ratio=0.99,
                local_hit_damage_visible=True,
            )
        )
        assert fsm.step(w_fight) == []
        assert fsm.state == BotState.COMBAT_WAIT

    # 4. 2.5s after last hit stops (t=23.1) -> MANDATORY BotState.LOOT -> RETURN_TO_CENTER -> SEARCH_MOB!
    fsm.step(tracker.update(_snap(23.1, x=645, y=1830, fw=1024, fh=768)))
    assert fsm.state == BotState.LOOT
    fsm.step(tracker.update(_snap(24.0, x=645, y=1830, fw=1024, fh=768)))
    assert fsm.state == BotState.LOOT  # inside mandatory 2.5s window
    fsm.step(tracker.update(_snap(25.5, x=645, y=1830, fw=1024, fh=768)))
    assert fsm.state == BotState.LOOT  # elapsed 2.4s < 2.5s mandatory min window
    fsm.step(tracker.update(_snap(25.7, x=645, y=1830, fw=1024, fh=768)))
    assert fsm.state == BotState.RETURN_TO_CENTER
    fsm.step(tracker.update(_snap(26.0, x=594, y=1832, fw=1024, fh=768)))
    assert fsm.state == BotState.SEARCH_MOB

    # 5. At drift=72u (outside 55u farm_radius, <=85u) with 2 visible mobs -> attacks mob instead of drifting/patrolling!
    spot = fsm.navigator.spot_center
    m_a = _mob(cx=710, cy=320, dist=180.0, has_hp=False)
    m_b = _mob(cx=300, cy=320, dist=240.0, has_hp=False)
    acts_atk = fsm.step(
        tracker.update(_snap(32.0, x=int(spot.x + 72), y=int(spot.y), fw=1024, fh=768, mobs=(m_a, m_b)))
    )
    assert fsm.state == BotState.COMBAT_WAIT
    assert any(isinstance(a, ClickScreenIntent) and a.x == 710 and a.y == 320 for a in acts_atk)


def test_ref_loot_bags_6_piles_detected_and_clicked_sequentially_in_loot():
    """Regression test for ref_loot_bags.jpg (1024x768, 6 gold piles at x=625..735, y=340..430):
    verifies all 6 piles are detected ONLY in BotState.LOOT and left-clicked sequentially via 22x22 visited_loot_grids."""
    from pathlib import Path
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = _load_legacy_mouse_loot_cfg()
    cfg.perception["hud_fixed_pixel_ui"] = True
    pipe = PerceptionPipeline(cfg)

    ref_path = Path(__file__).parent / "fixtures" / "ref_loot_bags.jpg"
    if ref_path.exists():
        img = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        assert img is not None
    else:
        # Calibrated 1024x768 synthetic ref_loot_bags frame: 6 gold piles lined up in x=625..735, y=340..430
        hsv_bg = np.zeros((768, 1024, 3), dtype=np.uint8)
        hsv_bg[:, :] = (28, 120, 62)
        img = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
        piles = [(632, 348), (658, 362), (684, 376), (710, 390), (656, 408), (696, 420)]
        for px, py in piles:
            # Dark-brown mound (12x12) + gold-orange gleam (HSV H=18, S=180, V=170) in center
            img[py : py + 12, px : px + 12] = cv2.cvtColor(
                np.full((12, 12, 3), (18, 120, 32), dtype=np.uint8), cv2.COLOR_HSV2BGR
            )
            img[py + 2 : py + 6, px + 3 : px + 8] = cv2.cvtColor(
                np.full((4, 5, 3), (18, 180, 170), dtype=np.uint8), cv2.COLOR_HSV2BGR
            )
        # Distractor inside SELF zone (x=500, y=380) & far outside 280px radius (x=140, y=420) -> MUST be rejected!
        for px, py in ((500, 380), (140, 420)):
            img[py : py + 12, px : px + 12] = cv2.cvtColor(
                np.full((12, 12, 3), (18, 120, 32), dtype=np.uint8), cv2.COLOR_HSV2BGR
            )
            img[py + 2 : py + 6, px + 3 : px + 8] = cv2.cvtColor(
                np.full((4, 5, 3), (18, 180, 170), dtype=np.uint8), cv2.COLOR_HSV2BGR
            )

    pkt = FramePacket(image_bgr=img, timestamp=1.0, source_name="ref_loot_bags.jpg")

    # 1. Hard-gate: GoldBagDetector MUST stay OFF in COMBAT_WAIT and SEARCH_MOB!
    snap_cw = pipe.analyze(pkt, current_state=BotState.COMBAT_WAIT, check_interface_mode=False)
    snap_scan = pipe.analyze(pkt, current_state=BotState.SEARCH_MOB, check_interface_mode=False)
    assert len(snap_cw.gold_bags) == 0 and len(snap_cw.mobs) == 0
    assert len(snap_scan.gold_bags) == 0

    # 2. In BotState.LOOT: detects all 6 gold piles inside <=280px (rejects SELF & far >280px distractors)!
    snap_loot = pipe.analyze(pkt, current_state=BotState.LOOT, check_interface_mode=False)
    assert len(snap_loot.gold_bags) == 6, f"Expected 6 gold piles in ref_loot_bags, got {len(snap_loot.gold_bags)}: {snap_loot.gold_bags}"
    assert len(snap_loot.mobs) == 0

    # 3. FSM clicks all 6 piles sequentially one-by-one using 22x22 visited_loot_grids (TTL 15s), zero DRAG_MMB!
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    fsm.transition(BotState.LOOT, 10.0, "post_kill_6_piles")

    clicked_pts: list[tuple[int, int]] = []
    for i in range(6):
        ts = 10.1 + i * 0.45
        w_l = tracker.update(
            PerceptionSnapshot(
                timestamp=ts,
                frame_width=1024,
                frame_height=768,
                hp_ratio=1.0,
                mp_ratio=1.0,
                coords=CoordReading(x=615, y=1830, confidence=0.98),
                minimap=MinimapReading(),
                loot=snap_loot.gold_bags,
            )
        )
        acts = fsm.step(w_l)
        assert fsm.state == BotState.LOOT
        assert not any(isinstance(a, DragCameraIntent) for a in acts)
        clicks = [a for a in acts if isinstance(a, ClickScreenIntent) and a.button == "left"]
        assert len(clicks) == 1, f"Expected 1 left-click on pile #{i + 1}, got {acts}"
        clicked_pts.append((clicks[0].x, clicks[0].y))

    assert len(set(clicked_pts)) == 6, f"All 6 piles must be clicked sequentially without repeats: {clicked_pts}"
    assert len(fsm.visited_loot_grids) == 6

    # 4. Once piles are gone for 1.2s (past min 2.0s window) -> RETURN_TO_CENTER!
    fsm.step(tracker.update(_snap(13.0, fw=1024, fh=768, loot=())))
    assert fsm.state == BotState.LOOT
    fsm.step(tracker.update(_snap(14.6, fw=1024, fh=768, loot=())))
    assert fsm.state == BotState.RETURN_TO_CENTER


def test_anti_ks_player_cyan_bar_external_damage_zone_wounded_hp_and_loot_avoidance():
    """Verify Anti-KS: cyan player HP bar (<=240px), external floating damage zone (<=220px, TTL 6.0s),
    wounded monster rejection (HP < 94%), and LOOT avoidance near other players / external damage zones."""
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = _load_legacy_mouse_loot_cfg()
    cfg.perception["hud_fixed_pixel_ui"] = True
    assert cfg.perception["anti_ks_enabled"] is True
    assert int(cfg.perception["anti_ks_player_radius_px"]) == 240
    assert int(cfg.perception["anti_ks_damage_zone_radius_px"]) == 220
    assert float(cfg.perception["anti_ks_damage_zone_ttl_sec"]) == 6.0
    assert float(cfg.perception["anti_ks_min_untouched_hp_ratio"]) == 0.94

    pipe = PerceptionPipeline(cfg)
    hsv_bg = np.zeros((768, 1024, 3), dtype=np.uint8)
    hsv_bg[:, :] = (28, 120, 62)

    # 1. Cyan player HP bar at (240, 320) + SELF cyan bar at (485, 380) + 2 mobs (1 within 240px of player, 1 far away)
    bgr_pl = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    cyan_bar = cv2.cvtColor(np.full((4, 58, 3), (94, 150, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    green_bar = cv2.cvtColor(np.full((4, 68, 3), (62, 210, 210), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr_pl[320:324, 211:269] = cyan_bar  # other player at cx=240, cy=322
    bgr_pl[380:384, 483:541] = cyan_bar  # SELF player inside x=435..590, y=305..465 -> ignored!
    bgr_pl[280:284, 300:368] = green_bar  # Mob near other player (~95px <= 240px -> CONTESTED!)
    bgr_pl[280:284, 740:808] = green_bar  # Far untouched mob (~534px > 240px -> SAFE!)

    snap_pl = pipe.analyze(
        FramePacket(image_bgr=bgr_pl, timestamp=1.0),
        current_state=BotState.SEARCH_MOB,
        check_interface_mode=False,
    )
    assert len(snap_pl.other_players) == 1 and abs(snap_pl.other_players[0][0] - 240) <= 6
    assert len(snap_pl.mobs) == 1 and abs(snap_pl.mobs[0].click_x - 774) <= 6
    assert len(snap_pl.contested_mobs) == 1 and abs(snap_pl.contested_mobs[0].click_x - 334) <= 6

    # 2. External floating damage numbers (stealth Sin hitting mob at (490, 260) while SELF is idle in SEARCH_MOB!)
    bgr_dmg = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    dmg_digit = cv2.cvtColor(np.full((22, 12, 3), (22, 160, 210), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr_dmg[245:267, 485:497] = dmg_digit  # floating '9' at (491, 256)
    snap_dmg = pipe.analyze(
        FramePacket(image_bgr=bgr_dmg, timestamp=2.0),
        current_state=BotState.SEARCH_MOB,
        check_interface_mode=False,
    )
    assert len(snap_dmg.external_damage_points) >= 1
    assert any(abs(dx - 491) <= 15 and abs(dy - 256) <= 15 for dx, dy in snap_dmg.external_damage_points)

    # 3. FSM Anti-KS: rejects mob under external damage zone (TTL 6.0s) AND rejects wounded mob (HP=0.85 < 0.94)!
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)

    mob_under_dmg = _mob(cx=500, cy=300, dist=140.0, has_hp=False)
    mob_wounded = _mob(cx=780, cy=310, dist=270.0, has_hp=True, hp=0.85)
    w_ks = tracker.update(
        PerceptionSnapshot(
            timestamp=5.0,
            frame_width=1024,
            frame_height=768,
            hp_ratio=1.0,
            mp_ratio=1.0,
            coords=CoordReading(x=615, y=1830, confidence=0.98),
            minimap=MinimapReading(),
            mobs=(mob_under_dmg, mob_wounded),
            external_damage_points=((491, 256),),
        )
    )
    acts_ks = fsm.step(w_ks)
    assert fsm.state == BotState.SEARCH_MOB, "Must NOT engage mob under external damage zone or wounded (<94%) mob!"
    assert not any(isinstance(a, ClickScreenIntent) for a in acts_ks)
    assert len(fsm.external_combat_zones) == 1

    # 4. Anti-KS in BotState.LOOT: skips GoldBag within <=220px of other_players / external_combat_zones, clicks safe bag!
    fsm.transition(BotState.LOOT, 6.0, "test_anti_ks_loot")
    bag_in_dmg = _bag(cx=510, cy=320, dist=120.0)   # ~67px from external damage zone (491, 256) -> SKIP!
    bag_near_pl = _bag(cx=290, cy=340, dist=225.0)  # ~53px from other player (240, 322) -> SKIP!
    bag_safe = _bag(cx=720, cy=410, dist=210.0)     # >220px from both -> CLICK!
    w_loot_ks = tracker.update(
        PerceptionSnapshot(
            timestamp=6.5,
            frame_width=1024,
            frame_height=768,
            hp_ratio=1.0,
            mp_ratio=1.0,
            coords=CoordReading(x=615, y=1830, confidence=0.98),
            minimap=MinimapReading(),
            other_players=((240, 322),),
            loot=(bag_in_dmg, bag_near_pl, bag_safe),
        )
    )
    acts_loot_ks = fsm.step(w_loot_ks)
    clicks = [a for a in acts_loot_ks if isinstance(a, ClickScreenIntent)]
    assert len(clicks) == 1 and clicks[0].x == 720 and clicks[0].y == 410


def test_ref_gold_scatter_10_piles_and_ref_close_mobs_regression():
    """Verify (1) ref_gold_scatter.jpg detects >=10 gold piles (both left & right) with 0 FP on grass/HUD/SELF,
    and (2) ref_close_mobs detects BOTH close Gold Rustlers (green HP bar next to SELF + right-side silhouette)."""
    from pathlib import Path
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = _load_legacy_mouse_loot_cfg()
    cfg.perception["hud_fixed_pixel_ui"] = True
    pipe = PerceptionPipeline(cfg)

    # 1. Real ref_gold_scatter.jpg regression (resized to 1024x768)
    scatter_path = Path(__file__).parent / "fixtures" / "ref_gold_scatter.jpg"
    assert scatter_path.exists(), f"Missing fixture: {scatter_path}"
    raw_sc = cv2.imread(str(scatter_path), cv2.IMREAD_COLOR)
    assert raw_sc is not None
    img_768 = cv2.resize(raw_sc, (1024, 768), interpolation=cv2.INTER_AREA)
    pkt_sc = FramePacket(image_bgr=img_768, timestamp=1.0, source_name="ref_gold_scatter.jpg")

    # Hard-gate: OFF in COMBAT_WAIT, RETURN_TO_CENTER, SEARCH_MOB, PAUSED; active EXCLUSIVELY in LOOT (<=310px)!
    assert pipe.analyze(pkt_sc, current_state=BotState.COMBAT_WAIT, check_interface_mode=False).gold_bags == ()
    assert pipe.analyze(pkt_sc, current_state=BotState.RETURN_TO_CENTER, check_interface_mode=False).gold_bags == ()
    assert pipe.analyze(pkt_sc, current_state=BotState.SEARCH_MOB, check_interface_mode=False).gold_bags == ()
    assert pipe.analyze(pkt_sc, current_state=BotState.PAUSED, check_interface_mode=False).gold_bags == ()
    snap_loot = pipe.analyze(pkt_sc, current_state=BotState.LOOT, check_interface_mode=False)
    assert len(snap_loot.gold_bags) >= 10, f"Expected >=10 gold piles on ref_gold_scatter.jpg, got {len(snap_loot.gold_bags)}: {snap_loot.gold_bags}"
    # Both left side (x < 450) and right side (x > 570) piles detected, zero inside SELF (468..556, 345..428) or chat!
    assert sum(1 for b in snap_loot.gold_bags if b.click_x < 450) >= 2
    assert sum(1 for b in snap_loot.gold_bags if b.click_x > 570) >= 8
    for b in snap_loot.gold_bags:
        assert not (468 <= b.click_x <= 556 and 345 <= b.click_y <= 428), f"FP in SELF: {b}"
        assert not (b.click_x >= 442 and b.click_y >= 518), f"FP in chat: {b}"

    # 2. Close-range Gold Rustler regression (ref_close_mobs.jpg or calibrated 1024x768 equivalent):
    #    - Left mob at x=300..475, y=355..565 with standalone green HP bar at x=352..444, y=394 (NO purple nameplate!)
    #    - Right mob at x=775..955, y=325..565 with metallic/dark-bronze silhouette (NO nameplate & NO HP bar!)
    #    - SELF cyan HP bar at x=480..544, y=385 -> ignored!
    close_path = Path(__file__).parent / "fixtures" / "ref_close_mobs.jpg"
    if close_path.exists():
        img_cm = cv2.imread(str(close_path), cv2.IMREAD_COLOR)
        assert img_cm is not None
    else:
        hsv_cm = np.zeros((768, 1024, 3), dtype=np.uint8)
        hsv_cm[:, :] = (32, 125, 92)  # daylight olive-green grass (med_v >= 72)
        img_cm = cv2.cvtColor(hsv_cm, cv2.COLOR_HSV2BGR)
        # SELF cyan HP bar inside tight SELF (472..552, 350..432)
        img_cm[385:389, 480:544] = cv2.cvtColor(np.full((4, 64, 3), (94, 150, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)
        # Left close mob standalone green HP bar at x=352..444, y=394 (92px wide, 5px tall, NO nameplate!)
        img_cm[394:399, 352:444] = cv2.cvtColor(np.full((5, 92, 3), (60, 200, 205), dtype=np.uint8), cv2.COLOR_HSV2BGR)
        # Right close mob silhouette at x=785..935, y=340..520 (dark bronze + metallic gray body, NO nameplate/HP bar)
        img_cm[340:520, 785:935] = cv2.cvtColor(np.full((180, 150, 3), (18, 70, 42), dtype=np.uint8), cv2.COLOR_HSV2BGR)
        img_cm[365:445, 815:905] = cv2.cvtColor(np.full((80, 90, 3), (110, 30, 125), dtype=np.uint8), cv2.COLOR_HSV2BGR)

    mobs_cm = pipe.mobs.detect(FramePacket(image_bgr=img_cm, timestamp=2.0, source_name="ref_close_mobs.jpg"))
    assert len(mobs_cm) >= 2, f"Expected >=2 close-range Gold Rustlers (HP bar + silhouette), got {len(mobs_cm)}: {mobs_cm}"
    assert any(300 <= m.click_x <= 475 and abs(m.click_y - 436) <= 10 for m in mobs_cm), f"Missed left HP-bar mob: {mobs_cm}"
    assert any(775 <= m.click_x <= 955 and 325 <= m.click_y <= 565 for m in mobs_cm), f"Missed right silhouette mob: {mobs_cm}"


def test_local_hit_damage_5s_no_hit_abort_and_5s_post_hit_loot_transition():
    """Verify (a) local_hit_damage_visible ignores faraway damage & 5.0s without hit aborts straight to SEARCH_MOB,
    and (b) confirmed local hit stays locked up to 5.0s after last hit then enters LOOT (10.0s)."""
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = load_config("config/default.yaml")
    cfg.perception["hud_fixed_pixel_ui"] = True
    pipe = PerceptionPipeline(cfg)

    hsv_bg = np.zeros((768, 1024, 3), dtype=np.uint8)
    hsv_bg[:, :] = (28, 120, 62)

    # Faraway damage number at x=180, y=260 (outside local x=340..685) -> local_hit_damage_visible MUST be False!
    bgr_far = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    dmg = cv2.cvtColor(np.full((18, 10, 3), (20, 200, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr_far[260:278, 180:190] = dmg
    snap_far = pipe.analyze(FramePacket(image_bgr=bgr_far, timestamp=1.0), current_state=BotState.COMBAT_WAIT, check_interface_mode=False)
    assert snap_far.local_hit_damage_visible is False

    # Near-character damage number at x=500, y=280 (inside x=340..685, y=190..465) -> local_hit_damage_visible MUST be True!
    bgr_near = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    bgr_near[280:298, 500:510] = dmg
    snap_near = pipe.analyze(FramePacket(image_bgr=bgr_near, timestamp=1.5), current_state=BotState.COMBAT_WAIT, check_interface_mode=False)
    assert snap_near.local_hit_damage_visible is True

    # (a) 5.0s without local_hit_damage_visible -> abort directly to SEARCH_MOB (NEVER LOOT) + blacklist 15s!
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    m1 = _mob(cx=320, cy=330, dist=180.0, has_hp=False)
    fsm.step(tracker.update(_snap(10.0, x=615, y=1830, fw=1024, fh=768, mobs=(m1,))))
    assert fsm.state == BotState.COMBAT_WAIT
    fsm.step(tracker.update(_snap(13.0, x=618, y=1832, fw=1024, fh=768)))
    assert fsm.state == BotState.COMBAT_WAIT
    fsm.step(tracker.update(_snap(15.1, x=619, y=1832, fw=1024, fh=768)))
    assert fsm.state == BotState.SEARCH_MOB, "Must abort straight to SEARCH_MOB after 5.0s without local hit!"

    # (b) With local_hit_damage_visible -> stays in COMBAT_WAIT until 5.0s after last hit, then enters LOOT (10.0s)!
    m2 = _mob(cx=720, cy=320, dist=200.0, has_hp=False)
    fsm.step(tracker.update(_snap(20.0, x=615, y=1830, fw=1024, fh=768, mobs=(m2,))))
    assert fsm.state == BotState.COMBAT_WAIT
    w_hit = tracker.update(
        PerceptionSnapshot(
            timestamp=22.0, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
            coords=CoordReading(x=615, y=1830, confidence=0.98), minimap=MinimapReading(),
            local_hit_damage_visible=True,
        )
    )
    assert fsm.step(w_hit) == [] and fsm.state == BotState.COMBAT_WAIT
    assert fsm.step(tracker.update(_snap(24.0, fw=1024, fh=768))) == [] and fsm.state == BotState.COMBAT_WAIT  # 2.0s < 2.5s
    fsm.step(tracker.update(_snap(24.6, fw=1024, fh=768)))  # 2.6s >= 2.5s since last local hit
    assert fsm.state == BotState.LOOT


def test_full_5_stage_cycle_combat_loot_10s_click_e_spam_return_to_center_594_1832():
    """Verify full 5-stage cycle with default config:
    1. AHK v2 background E spam (90ms + WinActive + chat_lock interlock around /setweather 11 & /kill).
    2. SCAN_SPOT (GoldBagDetector OFF) -> 1x click -> COMBAT_WAIT (all detectors OFF, 2.5s post-hit -> LOOT).
    3. LOOT (10.0s, GoldBagDetector EXCLUSIVELY ON, mouse aim + left-click gold + visited_loot_grids 20x20) -> RETURN_TO_CENTER.
    4. RETURN_TO_CENTER (594, 1832, arrival <=8u or 6.0s timeout, all object detectors OFF) -> SCAN_SPOT."""
    from pathlib import Path
    import cv2
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = load_config("config/default.yaml")
    cfg.perception["hud_fixed_pixel_ui"] = True
    assert cfg.farming["background_e_spam"] is True
    assert cfg.farming["disable_gold_bag_detector"] is False
    assert cfg.farming["loot_mouse_click_enabled"] is True
    assert float(cfg.farming["post_kill_loot_window_sec"]) == 8.0
    assert float(cfg.farming["combat_no_hit_abort_sec"]) == 5.0
    assert float(cfg.farming["combat_confirmed_max_sec"]) == 35.0
    assert float(cfg.farming["combat_post_last_hit_idle_sec"]) == 2.5
    assert int(cfg.farming["spot_x"]) == 594 and int(cfg.farming["spot_y"]) == 1832
    assert float(cfg.farming["spot_center_arrival_radius"]) == 12.0
    assert float(cfg.farming["return_center_timeout_sec"]) == 6.0

    # 1. AHK v2 bridge script checks & CHAT_CMD interlock
    ahk_src = Path("ahk/vikings_input_bridge.ahk").read_text(encoding="utf-8")
    for token in (
        "SetTimer(SpamPickupLoop, 90)",
        'WinActive("ahk_id "',
        'cmd = "SET_E_SPAM"',
        "chat_lock := true",
        "chat_lock := false",
        "SetTimer(SpamPickupLoop, 0)",
    ):
        assert token in ahk_src, f"Missing expected AHK v2 token: {token}"

    sent: list[str] = []
    drv = AhkInputDriver(
        ahk_mode="vm_foreground",
        command_transport=lambda cmd: (sent.append(cmd) or "OK"),
    )
    drv.set_e_spam(True)
    drv.send_chat_command("/setweather 11")
    drv.send_chat_command("/kill")
    drv.release_all()
    assert sent == [
        "SET_E_SPAM|0|1",
        "SET_E_SPAM|0|0",
        "CHAT_CMD|vm_foreground|0|/setweather 11",
        "SET_E_SPAM|0|1",
        "SET_E_SPAM|0|0",
        "CHAT_CMD|vm_foreground|0|/kill",
        "SET_E_SPAM|0|1",
        "SET_E_SPAM|0|0",
        "RELEASE_ALL|0",
    ]

    # 2. Pipeline: GoldBagDetector ON EXCLUSIVELY in BotState.LOOT, OFF in COMBAT_WAIT / RETURN_TO_CENTER / SEARCH_MOB
    pipe = PerceptionPipeline(cfg)
    scatter_path = Path(__file__).parent / "fixtures" / "ref_gold_scatter.jpg"
    img = cv2.resize(cv2.imread(str(scatter_path), cv2.IMREAD_COLOR), (1024, 768))
    pkt = FramePacket(image_bgr=img, timestamp=1.0)
    assert pipe.analyze(pkt, current_state=BotState.SEARCH_MOB, check_interface_mode=False).gold_bags == ()
    assert pipe.analyze(pkt, current_state=BotState.COMBAT_WAIT, check_interface_mode=False).gold_bags == ()
    snap_rtc = pipe.analyze(pkt, current_state=BotState.RETURN_TO_CENTER, check_interface_mode=False)
    assert snap_rtc.mobs == () and snap_rtc.gold_bags == () and snap_rtc.dead_corpses == ()
    snap_loot = pipe.analyze(pkt, current_state=BotState.LOOT, check_interface_mode=False)
    assert len(snap_loot.gold_bags) >= 1 and snap_loot.mobs == ()

    # 3. Full FSM 5-stage cycle: SCAN_SPOT -> COMBAT_WAIT -> LOOT (click gold) -> RETURN_TO_CENTER (594,1832) -> SCAN_SPOT
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    assert fsm.state == BotState.SEARCH_MOB

    # Stage 1 -> 2: Engage mob
    m1 = _mob(cx=720, cy=320, dist=200.0, has_hp=False)
    acts_eng = fsm.step(tracker.update(_snap(2.0, x=620, y=1840, fw=1024, fh=768, mobs=(m1,))))
    assert fsm.state == BotState.COMBAT_WAIT
    assert any(isinstance(a, ClickScreenIntent) and a.x == 720 and a.y == 320 for a in acts_eng)

    # Stage 2: Hit confirmed -> 2.5s post-hit idle -> LOOT
    w_hit = tracker.update(
        PerceptionSnapshot(
            timestamp=3.5, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
            coords=CoordReading(x=630, y=1845, confidence=0.98), minimap=MinimapReading(),
            local_hit_damage_visible=True,
        )
    )
    assert fsm.step(w_hit) == [] and fsm.state == BotState.COMBAT_WAIT
    assert fsm.step(tracker.update(_snap(6.1, x=630, y=1845, fw=1024, fh=768))) == []
    assert fsm.state == BotState.LOOT

    # Stage 3: LOOT clicks gold bag directly, stays >=2.5s min, then exits to RETURN_TO_CENTER once empty >= 1.2s
    bag1 = _bag(cx=640, cy=400, dist=130.0)
    acts_l1 = fsm.step(tracker.update(_snap(6.5, x=630, y=1845, fw=1024, fh=768, loot=(bag1,))))
    assert any(isinstance(a, ClickScreenIntent) and a.x == 640 and a.y == 400 and a.button == "left" for a in acts_l1)
    fsm.step(tracker.update(_snap(7.2, x=630, y=1845, fw=1024, fh=768, loot=())))
    assert fsm.state == BotState.LOOT  # elapsed 1.1s < 2.5s
    fsm.step(tracker.update(_snap(8.5, x=630, y=1845, fw=1024, fh=768, loot=())))
    assert fsm.state == BotState.LOOT  # elapsed 2.4s < 2.5s mandatory min window
    fsm.step(tracker.update(_snap(8.7, x=630, y=1845, fw=1024, fh=768, loot=())))
    assert fsm.state == BotState.RETURN_TO_CENTER

    # Stage 4: RETURN_TO_CENTER walks toward (594, 1832) while far (dist=38u > 12u), then transitions to SEARCH_MOB within <= 12u!
    acts_walk = fsm.step(tracker.update(_snap(9.2, x=630, y=1845, fw=1024, fh=768)))
    assert fsm.state == BotState.RETURN_TO_CENTER
    assert any(isinstance(a, ClickScreenIntent) and "return->spot_center" in a.reason for a in acts_walk)
    fsm.step(tracker.update(_snap(10.5, x=603, y=1837, fw=1024, fh=768)))  # dist = 10.3 <= 12.0
    assert fsm.state == BotState.SEARCH_MOB

    # Verify RETURN_TO_CENTER 6.0s timeout also returns cleanly to SEARCH_MOB
    fsm.transition(BotState.RETURN_TO_CENTER, 20.0, "test_rtc_timeout")
    fsm.step(tracker.update(_snap(26.2, x=630, y=1845, fw=1024, fh=768)))
    assert fsm.state == BotState.SEARCH_MOB


def test_yolo11n_onnx_detector_mock_session_filters_hard_gates_fallback_and_overlay():
    """Verify Yolo11nOnnxDetector with mock ONNX session (1, 8, 8400):
    - Letterbox 1024x768 -> 640x640, NMS (conf=0.45, iou=0.45), coordinate un-mapping
    - Spatial filters: SELF box (472..552, 350..432), HUD/chat (x>=442, y>=518, minimap, HP/MP, skill bar)
    - Class 3 dead_rustler ignored as attack target (overlay-only dead_corpses)
    - Anti-KS <= 240px from class 2 player & external_combat_zones rejects gold_rustler & gold_bag
    - Hard-gate: ZERO YOLO inference calls during COMBAT_WAIT and RETURN_TO_CENTER
    - Fallback to OpenCV-Gold when ONNX model file is missing
    - Overlay displays YOLO class labels + confidence scores
    """
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.debug_overlay import build_overlay_frame_data, draw_overlay_scene
    from vikings_bot.perception.pipeline import PerceptionPipeline
    from vikings_bot.perception.yolo_detector import (
        CLS_DEAD_RUSTLER,
        CLS_GOLD_BAG,
        CLS_GOLD_RUSTLER,
        CLS_PLAYER,
        Yolo11nOnnxDetector,
        letterbox_bgr,
    )
    from vikings_bot.runtime.overlay import DebugOverlay

    cfg = load_config("config/default.yaml")
    assert cfg.perception["yolo_enabled"] is True
    assert cfg.perception["yolo_model_path"] == "models/yolo11n_ryl.onnx"
    assert int(cfg.perception["yolo_imgsz"]) == 640
    assert float(cfg.perception["yolo_conf_threshold"]) == 0.45
    assert float(cfg.perception["gold_bag_conf_threshold"]) == 0.22
    assert cfg.perception["loot_tiled_enabled"] is True
    assert float(cfg.perception["loot_tiled_gold_bag_conf_threshold"]) == 0.22
    assert float(cfg.perception["dead_rustler_conf_threshold"]) == 0.35
    # Test single-frame path here; 3-crop tiled path is verified in test_yolo11n_detect_loot_tiled_3_crops_merge_nms_and_fsm_chase
    cfg.perception["loot_tiled_enabled"] = False
    assert float(cfg.perception["yolo_click_y_ratio"]) == 0.68
    assert float(cfg.perception["yolo_iou_threshold"]) == 0.45
    assert int(cfg.perception["yolo_intra_op_threads"]) == 2
    assert float(cfg.perception["paused_yolo_preview_hz"]) == 2.5
    assert cfg.perception["anti_ks_use_hsv_damage"] is False
    assert float(cfg.perception["anti_ks_player_min_conf"]) == 0.50
    assert float(cfg.farming["combat_min_lock_hold_sec"]) == 2.0
    assert float(cfg.farming["engage_click_hold_sec"]) == 0.12
    assert cfg.farming["engage_skill_key"] == "1"
    assert float(cfg.farming["return_center_min_drift_units"]) == 35.0
    assert float(cfg.farming["loot_min_window_sec"]) == 2.5

    # Helper: map original 1024x768 xywh -> 640x640 letterbox xywh
    # Draw bright orange HSV-damage-like patch at (755, 285) right next to safe gold_rustler -> MUST NOT cause KS:DMG!
    import cv2

    blank = np.full((768, 1024, 3), 40, dtype=np.uint8)
    blank[275:295, 750:764] = cv2.cvtColor(np.full((20, 14, 3), (20, 220, 230), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    blob_chk, r_lb, pad_x, pad_y = letterbox_bgr(blank, new_shape=640)
    assert blob_chk.shape == (1, 3, 640, 640)
    assert blob_chk.dtype == np.float32
    assert 0.0 <= float(blob_chk.min()) <= float(blob_chk.max()) <= 1.0

    def _to_lb(ox: float, oy: float, ow: float, oh: float) -> tuple[float, float, float, float]:
        return (ox * r_lb + pad_x, oy * r_lb + pad_y, ow * r_lb, oh * r_lb)

    # Build (1, 8, 8400) mock output tensor
    tensor = np.zeros((1, 8, 8400), dtype=np.float32)
    entries = [
        # 0: Safe gold_rustler at (760, 290) (far from player at 200, 300 -> dist=560 > 240; orange HSV patch ignored!)
        (CLS_GOLD_RUSTLER, 0.88, 760, 290, 72, 92),
        # 1: Duplicate overlapping gold_rustler at (764, 292) lower conf -> suppressed by NMS!
        (CLS_GOLD_RUSTLER, 0.62, 764, 292, 70, 90),
        # 2: Low-conf gold_rustler (0.35 < 0.45) -> filtered out!
        (CLS_GOLD_RUSTLER, 0.35, 680, 260, 60, 80),
        # 3: gold_rustler inside SELF box (510, 390) -> filtered out!
        (CLS_GOLD_RUSTLER, 0.91, 510, 390, 60, 70),
        # 4: gold_rustler inside chat HUD (600, 620) -> filtered out!
        (CLS_GOLD_RUSTLER, 0.93, 600, 620, 64, 80),
        # 5: gold_rustler near high-conf player (280, 310, dist=80.6 <= 240) -> Anti-KS contested!
        (CLS_GOLD_RUSTLER, 0.84, 280, 310, 68, 86),
        # 6: High-conf player at (200, 300) (conf=0.82 >= 0.50 -> active Anti-KS!)
        (CLS_PLAYER, 0.82, 200, 300, 56, 88),
        # 7: Low-conf player at (860, 300) (conf=0.47 < 0.50, dist=100<=240 -> MUST NOT trigger Anti-KS on safe gold_rustler at 760,290!)
        (CLS_PLAYER, 0.47, 860, 300, 50, 80),
        # 8: dead_rustler at (620, 340) with conf=0.38 (>= 0.35 dead_rustler_conf_threshold) -> overlay dead_corpses only!
        (CLS_DEAD_RUSTLER, 0.38, 620, 340, 82, 48),
        # 9: Safe gold_bag at (630, 380) (conf=0.90)
        (CLS_GOLD_BAG, 0.90, 630, 380, 22, 20),
        # 10: Small gold_bag at (660, 390) with conf=0.27 (>= 0.25 gold_bag_conf_threshold, < 0.45!) -> MUST BE DETECTED!
        (CLS_GOLD_BAG, 0.27, 660, 390, 14, 14),
        # 11: Far gold_bag at (900, 200) (dist from 512,384 = 429 > 310) -> rejected in LOOT!
        (CLS_GOLD_BAG, 0.87, 900, 200, 20, 18),
        # 12: Anti-KS gold_bag near high-conf player at (250, 350, dist to player=70.7 <= 240) -> rejected!
        (CLS_GOLD_BAG, 0.85, 250, 350, 20, 18),
        # 13: gold_bag near external_combat_zone at (600, 250) (far from player, dist=403>240) -> NOT rejected because anti_ks_use_hsv_damage=False!
        (CLS_GOLD_BAG, 0.86, 600, 250, 20, 18),
    ]
    for idx, (cid, conf, ox, oy, ow, oh) in enumerate(entries):
        lx, ly, lw, lh = _to_lb(ox, oy, ow, oh)
        tensor[0, 0, idx] = lx
        tensor[0, 1, idx] = ly
        tensor[0, 2, idx] = lw
        tensor[0, 3, idx] = lh
        tensor[0, 4 + cid, idx] = conf

    class _MockIo:
        def __init__(self, name: str, shape=None) -> None:
            self.name = name
            self.shape = shape

    class _MockOrtSession:
        def __init__(self, out_arr: np.ndarray, imgsz: int = 640) -> None:
            self.out_arr = out_arr
            self.imgsz = imgsz
            self.calls = 0

        def get_inputs(self):
            return [_MockIo("images", shape=[1, 3, self.imgsz, self.imgsz])]

        def get_outputs(self):
            return [_MockIo("output0", shape=list(self.out_arr.shape))]

        def run(self, output_names, feed_dict):
            self.calls += 1
            assert "images" in feed_dict and feed_dict["images"].shape == (1, 3, self.imgsz, self.imgsz)
            return [self.out_arr]

    mock_sess = _MockOrtSession(tensor)
    pipe = PerceptionPipeline(cfg, yolo_session_override=mock_sess)
    assert pipe.active_backend_desc == "YOLO11n-ONNX"

    pkt = FramePacket(image_bgr=blank, timestamp=1.0)

    # 1. SCAN_SPOT (SEARCH_MOB): 1 safe gold_rustler, 1 contested (near player), 1 player, 1 dead_rustler, 0 loot
    snap_scan = pipe.analyze(
        pkt,
        current_state=BotState.SEARCH_MOB,
        check_interface_mode=False,
        external_combat_zones=((415.0, 255.0),),
    )
    assert mock_sess.calls == 1
    assert snap_scan.detector_backend == "YOLO11n-ONNX"
    assert len(snap_scan.mobs) == 1
    assert abs(snap_scan.mobs[0].click_x - 760) <= 3
    # Verify hitbox click_y = y1 + 0.68 * (y2 - y1)
    mb = snap_scan.mobs[0].body_bbox
    expected_cy = int(round(mb[1] + 0.68 * (mb[3] - mb[1])))
    assert abs(snap_scan.mobs[0].click_y - expected_cy) <= 1
    assert snap_scan.mobs[0].class_name == "gold_rustler"

    # 1b. PAUSED preview runs YOLO for gold_rustler/player/dead_rustler (loot stays OFF)
    snap_paused = pipe.analyze(pkt, current_state=BotState.PAUSED, check_interface_mode=False)
    assert len(snap_paused.mobs) == 1 and snap_paused.loot == ()

    # 1c. ENGAGE_MOB sends 120ms click-hold + confirm click + skill '1'
    tracker_e = WorldStateTracker()
    fsm_e = FarmingFSM(cfg, tracker_e, mode=BotMode.FULL)
    fsm_e.farm.auto_daylight_weather = False
    fsm_e.farm.auto_camera_pitch = False
    fsm_e.resume(1.0, ensure_mouse_mode=False)
    acts_eng_e = fsm_e.step(tracker_e.update(snap_scan))
    assert fsm_e.state == BotState.COMBAT_WAIT
    assert any(isinstance(a, ClickScreenIntent) and abs(a.hold_sec - 0.12) < 1e-4 for a in acts_eng_e)
    assert any(isinstance(a, TapKeyIntent) and a.key == "1" for a in acts_eng_e)
    assert abs(snap_scan.mobs[0].confidence - 0.88) < 1e-4
    assert len(snap_scan.contested_mobs) == 1 and "KS:PLAYER" in snap_scan.contested_mobs[0].contest_reason
    assert all("KS:DMG" not in m.contest_reason for m in snap_scan.contested_mobs)
    assert snap_scan.external_damage_points == ()
    assert len(snap_scan.other_players) == 1
    assert len(snap_scan.other_player_boxes) == 2
    assert len(snap_scan.dead_corpses) == 1
    assert snap_scan.loot == ()

    # 2. COMBAT_WAIT & RETURN_TO_CENTER: HARD-GATE -> ZERO YOLO inference calls!
    calls_before_gate = mock_sess.calls
    snap_cw = pipe.analyze(pkt, current_state=BotState.COMBAT_WAIT, check_interface_mode=False)
    snap_rtc = pipe.analyze(pkt, current_state=BotState.RETURN_TO_CENTER, check_interface_mode=False)
    assert mock_sess.calls == calls_before_gate, "YOLO MUST NOT run during COMBAT_WAIT or RETURN_TO_CENTER!"
    assert snap_cw.mobs == () and snap_cw.loot == () and snap_cw.dead_corpses == ()
    assert snap_rtc.mobs == () and snap_rtc.loot == () and snap_rtc.dead_corpses == ()

    # 3. LOOT (<= 310px): 3 safe gold_bags (0.90 at 630,380; small 0.35 at 660,390; 0.86 at 600,250);
    #    far (>310px) & player-KS (<=240px from conf>=0.50 player) bags rejected; 0 mobs
    snap_loot = pipe.analyze(
        pkt,
        current_state=BotState.LOOT,
        check_interface_mode=False,
        external_combat_zones=((600.0, 250.0),),
    )
    assert mock_sess.calls == calls_before_gate + 1
    assert snap_loot.mobs == ()
    assert len(snap_loot.loot) == 3
    assert any(abs(b.click_x - 630) <= 3 and b.class_name == "gold_bag" and abs(b.score - 0.90) < 1e-4 for b in snap_loot.loot)
    assert any(abs(b.click_x - 660) <= 3 and b.class_name == "gold_bag" and abs(b.score - 0.27) < 1e-4 for b in snap_loot.loot)
    assert any(abs(b.click_x - 600) <= 3 and b.class_name == "gold_bag" for b in snap_loot.loot)

    # 3b. Dynamic 416x416 ONNX input shape auto-sync (no hardcoded 640!)
    tensor_416 = np.zeros((1, 8, 3549), dtype=np.float32)
    mock_416 = _MockOrtSession(tensor_416, imgsz=416)
    det_416 = Yolo11nOnnxDetector(cfg, session_override=mock_416)
    assert det_416.imgsz_h == 416 and det_416.imgsz_w == 416
    det_416.detect(blank)
    assert mock_416.calls == 1

    # 3c. Mandatory >=2.5s LOOT clicks dead_rustler even when GOLD=0; when within <=35u of (594,1832) exits directly to SEARCH_MOB!
    tracker_l = WorldStateTracker()
    fsm_l = FarmingFSM(cfg, tracker_l, mode=BotMode.FULL)
    fsm_l.resume(1.0, ensure_mouse_mode=False)
    fsm_l.transition(BotState.LOOT, 10.0, "post_kill_zero_gold")
    w_dead = tracker_l.update(
        PerceptionSnapshot(
            timestamp=10.5, frame_width=1024, frame_height=768, hp_ratio=1.0, mp_ratio=1.0,
            coords=CoordReading(x=598, y=1835, confidence=0.98), minimap=MinimapReading(),
            loot=(), dead_corpses=((579, 316, 661, 364),),
        )
    )
    acts_dead = fsm_l.step(w_dead)
    assert fsm_l.state == BotState.LOOT
    assert any(isinstance(a, ClickScreenIntent) and a.x == 620 and a.y == 340 and "dead_rustler" in a.reason for a in acts_dead)
    # After >=2.5s empty & dist=5u <= 35u -> skips unnecessary RETURN_TO_CENTER and goes straight to SEARCH_MOB!
    fsm_l.step(tracker_l.update(_snap(12.7, x=598, y=1835, fw=1024, fh=768, loot=())))
    assert fsm_l.state == BotState.SEARCH_MOB

    # 4. Overlay displays YOLO class names + confidence scores
    tracker = WorldStateTracker()
    world_scan = tracker.update(snap_scan)
    ov_data = build_overlay_frame_data(
        game_rect=(0, 0, 1024, 768),
        world=world_scan,
        fsm_state=BotState.SEARCH_MOB,
        config=cfg,
        loop_hz=20.0,
    )
    drawn_texts: list[str] = []

    class _FakeCanvas:
        def delete(self, *_a, **_k):
            pass

        def create_rectangle(self, *_a, **_k):
            pass

        def create_oval(self, *_a, **_k):
            pass

        def create_line(self, *_a, **_k):
            pass

        def create_text(self, *_a, **kw):
            if "text" in kw:
                drawn_texts.append(str(kw["text"]))

    draw_overlay_scene(_FakeCanvas(), ov_data)
    assert any("gold_rustler 0.88" in t for t in drawn_texts), f"Missing gold_rustler conf label: {drawn_texts}"
    assert any("player 0.82" in t for t in drawn_texts), f"Missing player conf label: {drawn_texts}"
    assert any("dead_rustler 0.38" in t for t in drawn_texts), f"Missing dead_rustler conf label: {drawn_texts}"

    cv_ov = DebugOverlay(cfg).render(pkt, world_scan, BotState.SEARCH_MOB)
    assert cv_ov.shape == (768, 1024, 3)

    # 5. Fallback to OpenCV-Gold when ONNX model file is missing
    cfg_missing = load_config("config/default.yaml")
    cfg_missing.perception["yolo_model_path"] = "models/nonexistent_yolo11n_ryl.onnx"
    det_missing = Yolo11nOnnxDetector(cfg_missing)
    assert det_missing.is_available is False
    pipe_fb = PerceptionPipeline(cfg_missing)
    assert pipe_fb.active_backend_desc == "OpenCV-Gold"


def test_yolo11n_detect_loot_tiled_3_crops_merge_nms_and_fsm_chase():
    """Verify 3-crop 512x512 overlapping tiled LOOT detection (y=140..652), full-frame coordinate mapping,
    class-wise NMS (iou=0.45), conf=0.22, SELF/HUD/Anti-KS filtering, overlay rendering, and FSM chase
    (8.0s max, 24x24 grid, 1.4s move timeout, 0.8s empty exit, dead_rustler 1x click)."""
    import numpy as np
    from vikings_bot.capture.screen import FramePacket
    from vikings_bot.debug_overlay import build_overlay_frame_data, draw_overlay_scene
    from vikings_bot.perception.pipeline import PerceptionPipeline
    from vikings_bot.perception.yolo_detector import (
        CLS_DEAD_RUSTLER,
        CLS_GOLD_BAG,
        CLS_PLAYER,
        letterbox_bgr,
    )

    cfg = load_config("config/default.yaml")
    assert cfg.perception["loot_tiled_enabled"] is True
    assert float(cfg.perception["loot_tiled_gold_bag_conf_threshold"]) == 0.22
    assert float(cfg.farming["post_kill_loot_window_sec"]) == 8.0
    assert int(cfg.farming["loot_attempt_grid_px"]) == 24
    assert float(cfg.farming["loot_empty_exit_sec"]) == 0.8
    assert float(cfg.farming["loot_target_move_timeout_sec"]) == 1.4

    # 512x512 tile -> 640x640 letterbox ratio = 640 / 512 = 1.25, pad = 0
    tile_dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    _b, r_tile, px_tile, py_tile = letterbox_bgr(tile_dummy, new_shape=640)
    assert abs(r_tile - 1.25) < 1e-4 and px_tile == 0.0 and py_tile == 0.0

    def _tile_tensor(items: list[tuple[int, float, float, float, float, float]]) -> np.ndarray:
        arr = np.zeros((1, 8, 8400), dtype=np.float32)
        for i, (cid, conf, tx, ty, tw, th) in enumerate(items):
            arr[0, 0, i] = tx * r_tile + px_tile
            arr[0, 1, i] = ty * r_tile + py_tile
            arr[0, 2, i] = tw * r_tile
            arr[0, 3, i] = th * r_tile
            arr[0, 4 + cid, i] = conf
        return arr

    # Tile 0 (Left: x=0..512, y=140..652):
    #   - Overlapping gold_bag at full (430, 360) -> tile (430, 220), conf=0.60 (dist to player 150,340 = 281 > 240)
    #   - Player at full (150, 340) -> tile (150, 200), conf=0.85
    #   - Anti-KS gold_bag near player at full (210, 350) -> tile (210, 210), conf=0.80 (dist to player=61 <= 240 -> rejected!)
    t_left = _tile_tensor(
        [
            (CLS_GOLD_BAG, 0.60, 430, 220, 20, 18),
            (CLS_PLAYER, 0.85, 150, 200, 54, 86),
            (CLS_GOLD_BAG, 0.80, 210, 210, 20, 18),
        ]
    )
    # Tile 1 (Center: x=256..768, y=140..652):
    #   - Same overlapping gold_bag at full (432, 360) -> tile (176, 220), conf=0.88 (higher conf -> wins NMS!)
    #   - SELF gold_bag at full (512, 390) -> tile (256, 250), conf=0.92 (rejected by SELF box!)
    #   - HUD chat gold_bag at full (600, 580) -> tile (344, 440), conf=0.91 (rejected by HUD/chat!)
    #   - dead_rustler at full (620, 340) -> tile (364, 200), conf=0.42
    t_center = _tile_tensor(
        [
            (CLS_GOLD_BAG, 0.88, 176, 220, 20, 18),
            (CLS_GOLD_BAG, 0.92, 256, 250, 18, 18),
            (CLS_GOLD_BAG, 0.91, 344, 440, 20, 18),
            (CLS_DEAD_RUSTLER, 0.42, 364, 200, 76, 48),
        ]
    )
    # Tile 2 (Right: x=512..1024, y=140..652):
    #   - Desynced far-right gold_bag at full (880, 380) -> tile (368, 240), conf=0.23 (>= 0.22 threshold -> MUST BE DETECTED!)
    t_right = _tile_tensor([(CLS_GOLD_BAG, 0.23, 368, 240, 16, 16)])

    class _Io:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class _TiledMockSession:
        def __init__(self):
            self.outputs = [t_left, t_center, t_right]
            self.calls = 0

        def get_inputs(self):
            return [_Io("images", [1, 3, 640, 640])]

        def get_outputs(self):
            return [_Io("output0", [1, 8, 8400])]

        def run(self, _out_names, feed_dict):
            assert feed_dict["images"].shape == (1, 3, 640, 640)
            out = self.outputs[self.calls % 3]
            self.calls += 1
            return [out]

    sess = _TiledMockSession()
    pipe = PerceptionPipeline(cfg, yolo_session_override=sess)
    blank = np.full((768, 1024, 3), 40, dtype=np.uint8)
    pkt = FramePacket(image_bgr=blank, timestamp=1.0)

    snap_loot = pipe.analyze(pkt, current_state=BotState.LOOT, check_interface_mode=False)
    assert sess.calls == 3, f"Expected exactly 3 tiled ONNX calls in LOOT, got {sess.calls}"
    assert snap_loot.mobs == ()
    assert len(snap_loot.loot) == 2, f"Expected 2 merged unique gold_bags (NMS + SELF/HUD/Anti-KS), got {snap_loot.loot}"
    # Bag #1: merged overlap at ~(432, 360) with winning conf=0.88 (closest to 512,384)
    assert abs(snap_loot.loot[0].click_x - 432) <= 3 and abs(snap_loot.loot[0].score - 0.88) < 1e-4
    # Bag #2: desynced far-right at ~(880, 380) with conf=0.23 (>= 0.22)
    assert abs(snap_loot.loot[1].click_x - 880) <= 3 and abs(snap_loot.loot[1].score - 0.23) < 1e-4
    assert len(snap_loot.dead_corpses) == 1

    # Verify overlay renders tiled gold_bag boxes + conf scores during LOOT
    tracker = WorldStateTracker()
    w_loot = tracker.update(snap_loot)
    ov = build_overlay_frame_data(game_rect=(0, 0, 1024, 768), world=w_loot, fsm_state=BotState.LOOT, config=cfg, loop_hz=20.0)
    assert len(ov.loot) == 2
    texts: list[str] = []

    class _Canvas:
        def delete(self, *_a, **_k): pass
        def create_rectangle(self, *_a, **_k): pass
        def create_oval(self, *_a, **_k): pass
        def create_line(self, *_a, **_k): pass
        def create_text(self, *_a, **kw):
            if "text" in kw:
                texts.append(str(kw["text"]))

    draw_overlay_scene(_Canvas(), ov)
    assert any("gold_bag 0.88" in t for t in texts) and any("gold_bag 0.23" in t for t in texts)

    # Verify FSM LOOT 1.4s move timeout marks stuck grid visited & advances to next gold_bag, then exits after 0.8s empty!
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)
    fsm.transition(BotState.LOOT, 10.0, "test_tiled_chase")
    # Tick 1: clicks nearest bag (432, 360)
    acts1 = fsm.step(tracker.update(_snap(10.1, x=640, y=1840, fw=1024, fh=768, loot=snap_loot.loot)))
    assert any(isinstance(a, ClickScreenIntent) and abs(a.x - 432) <= 3 for a in acts1)
    # After 1.4s moving toward (432, 360) -> grid force-marked visited -> immediately clicks next bag (880, 380)!
    acts2 = fsm.step(tracker.update(_snap(11.6, x=640, y=1840, fw=1024, fh=768, loot=snap_loot.loot)))
    assert any(isinstance(a, ClickScreenIntent) and abs(a.x - 880) <= 3 for a in acts2)
    # Bags cleared for >=0.8s (past 2.5s min) -> exits to RETURN_TO_CENTER (dist=46.7 > 35u)!
    fsm.step(tracker.update(_snap(12.2, x=640, y=1840, fw=1024, fh=768, loot=())))
    assert fsm.state == BotState.LOOT
    fsm.step(tracker.update(_snap(13.1, x=640, y=1840, fw=1024, fh=768, loot=())))
    assert fsm.state == BotState.RETURN_TO_CENTER
