"""Unit tests for Mouse-Mode 1-click combat, Auto-ESC guard, Anti-Miss approach guard, 3-retry loot cap & AHK v2 bridge."""

from __future__ import annotations

from vikings_bot.brain.fsm import FarmingFSM
from vikings_bot.capture.window import GameWindow
from vikings_bot.cli import build_parser
from vikings_bot.config import load_config
from vikings_bot.input.actions import ClickScreenIntent, DragCameraIntent, TapKeyIntent
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
    )


def test_mouse_mode_single_click_engage_zero_spam_until_kill_then_loot_and_mmb_sweep():
    cfg = load_config("config/default.yaml")
    assert cfg.controls["mode"] == "mouse"
    assert cfg.controls["input_backend"] == "ahk"
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0)
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
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0)

    # Top banner shows INTERFACE TYPE: Keyboard -> presses ESC once (no ping-pong during cooldown!)
    acts_esc = fsm.step(tracker.update(_snap(1.2, interface_mode="keyboard_banner")))
    assert len(acts_esc) == 1
    assert isinstance(acts_esc[0], TapKeyIntent) and acts_esc[0].key == "esc"

    # Second tick immediately after -> respects 2.2s cooldown & waits cleanly
    acts_wait = fsm.step(tracker.update(_snap(1.4, interface_mode="keyboard_banner")))
    assert acts_wait == []

    # Mouse banner confirmed -> resumes searching & clicking normally
    m = _mob(cx=980, cy=450, dist=160.0)
    acts_ok = fsm.step(tracker.update(_snap(1.7, mobs=(m,), interface_mode="mouse_banner")))
    assert fsm.state == BotState.COMBAT_WAIT
    assert any(isinstance(a, ClickScreenIntent) for a in acts_ok)


def test_watchdog_anti_miss_never_cancels_while_distance_is_actively_shrinking():
    """Refinement #1: Watchdog 2.4s MUST NOT cancel lock while dist_to_player_px keeps decreasing."""
    cfg = load_config("config/default.yaml")
    tracker = WorldStateTracker()
    fsm = FarmingFSM(cfg, tracker, mode=BotMode.FULL)
    fsm.resume(1.0)

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
    fsm.resume(1.0)

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
    fsm.resume(1.0)

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
    fsm2.resume(10.0)
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
    fsm3.resume(20.0)
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
    fsm.resume(1.0)

    # 1. No kill yet + faraway yellow drop (dist=290px > 220px) -> MUST stay in SEARCH_MOB (never detour to LOOT!)
    far_drop = _bag(cx=820, cy=480, dist=290.0)
    fsm.step(tracker.update(_snap(1.5, fw=1024, fh=768, loot=(far_drop,))))
    assert fsm.state == BotState.SEARCH_MOB

    # 2. Record a kill -> within 4.0s window, even a 290px drop IS allowed to enter LOOT!
    tracker.record_kill_pos((820, 480), now=2.0)
    fsm.step(tracker.update(_snap(4.5, fw=1024, fh=768, loot=(far_drop,))))
    assert fsm.state == BotState.LOOT

    # 3. After > 4.0s with no kill, close bag (<=220px) enters LOOT and jitters by +-4px across 3 attempts ->
    #    quantized to 24x24px grid so 3rd attempt ALWAYS triggers blacklist_point!
    tracker2 = WorldStateTracker()
    fsm2 = FarmingFSM(cfg, tracker2, mode=BotMode.FULL)
    fsm2.resume(10.0)
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
