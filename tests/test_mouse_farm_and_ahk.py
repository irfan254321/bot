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
    )


def test_mouse_mode_single_click_engage_zero_spam_until_kill_then_loot_and_mmb_sweep():
    cfg = load_config("config/default.yaml")
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

    # 6. Bags cleared (4 empty frames) -> returns to SEARCH_MOB -> empty playfield rotates camera via MMB drag!
    for i in range(4):
        fsm.step(tracker.update(_snap(5.0 + i * 0.1, mobs=(), loot=())))
    assert fsm.state == BotState.SEARCH_MOB

    acts_sweep = fsm.step(tracker.update(_snap(5.6, mobs=(), loot=())))
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
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)

    # Simulate 80m repair timer already due while an obstructed bag is sitting on the ground
    fsm._active_farming_sec = 4810.0
    blocked_bag = _bag(cx=1120, cy=520, dist=165.0)

    # Enters LOOT first
    fsm.step(tracker.update(_snap(1.5, loot=(blocked_bag,))))
    assert fsm.state == BotState.LOOT

    # Attempt 1, 2, 3 (spaced past click cooldown 0.42s) -> on attempt #3 blacklists coord & exits LOOT!
    blacklisted = None
    for ts in (2.0, 2.6, 3.2):
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
    """Verify LOOT only triggers within 4.0s of kill or <=220px from feet, and 24x24px grid blacklists jittery bags on try #3."""
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0, ensure_mouse_mode=False)

    # 1. No kill yet + faraway yellow drop (dist=290px > 220px) -> MUST stay in SEARCH_MOB (never detour to LOOT!)
    far_drop = _bag(cx=820, cy=480, dist=290.0)
    fsm.step(tracker.update(_snap(1.5, fw=1024, fh=768, loot=(far_drop,))))
    assert fsm.state == BotState.SEARCH_MOB

    # 2. Record a kill -> within 4.0s window, near_last_kill drop IS allowed to enter LOOT!
    tracker.record_kill_pos((820, 480), now=2.0)
    fsm.step(tracker.update(_snap(2.4, fw=1024, fh=768, loot=(far_drop,))))
    assert fsm.state == BotState.LOOT

    # 3. After > 4.0s with no kill, close bag (<=220px) enters LOOT and jitters by +-4px across 3 attempts ->
    #    quantized to 24x24px grid so 3rd attempt ALWAYS triggers blacklist_point!
    tracker2 = WorldStateTracker()
    fsm2 = FarmingFSM(cfg, tracker2, mode=BotMode.FULL)
    fsm2.resume(10.0, ensure_mouse_mode=False)
    jitter_bags = [
        _bag(cx=510, cy=500, dist=180.0),
        _bag(cx=516, cy=505, dist=182.0),
        _bag(cx=513, cy=503, dist=181.0),
    ]
    fsm2.step(tracker2.update(_snap(10.5, fw=1024, fh=768, loot=(jitter_bags[0],))))
    assert fsm2.state == BotState.LOOT

    blacklisted = None
    for idx, ts in enumerate((11.0, 11.6, 12.2)):
        fsm2.step(tracker2.update(_snap(ts, fw=1024, fh=768, loot=(jitter_bags[idx],))))
        if fsm2.last_blacklisted_loot is not None:
            blacklisted = fsm2.last_blacklisted_loot
    assert blacklisted is not None
    assert fsm2.state == BotState.SEARCH_MOB


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

    # After 1.5s with no movement & no THP -> Anti-Ghost-Lock cancels COMBAT_WAIT and blacklists (380, 310) for 15s!
    fsm3.step(tracker3.update(_snap(21.7, x=615, y=1830, fw=1024, fh=768, mobs=())))
    assert fsm3.state == BotState.SEARCH_MOB
    acts_skip = fsm3.step(tracker3.update(_snap(25.0, x=615, y=1830, fw=1024, fh=768, mobs=(ghost,))))
    assert fsm3.state == BotState.SEARCH_MOB and not any(isinstance(a, ClickScreenIntent) for a in acts_skip)
