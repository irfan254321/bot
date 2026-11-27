"""Tests for Win32 Transparent & Click-Through Debug Overlay and BotEngine/CLI integration."""

from __future__ import annotations

from ctypes import wintypes
from typing import Any

from vikings_bot.cli import build_parser
from vikings_bot.config import load_config
from vikings_bot.debug_overlay import (
    CLICKTHROUGH_EXSTYLE_MASK,
    GA_ROOT,
    GWL_EXSTYLE,
    GWLP_HWNDPARENT,
    HWND_TOPMOST,
    LWA_COLORKEY,
    SW_HIDE,
    SW_SHOWNOACTIVATE,
    SWP_NOACTIVATE,
    SWP_NOOWNERZORDER,
    SWP_SHOWWINDOW,
    TRANSPARENT_COLOR_BGR,
    WDA_EXCLUDEFROMCAPTURE,
    WS_EX_LAYERED,
    WS_EX_NOACTIVATE,
    WS_EX_TOPMOST,
    WS_EX_TRANSPARENT,
    TransparentDebugOverlay,
    apply_clickthrough_styles,
    build_overlay_frame_data,
    compute_self_exclusion_boxes,
    draw_overlay_scene,
    position_overlay_window,
)
from vikings_bot.runtime.engine import BotEngine
from vikings_bot.runtime.safety import SafetyMonitor
from vikings_bot.state.models import (
    BotMode,
    BotState,
    MinimapReading,
    PerceptionSnapshot,
    Point2D,
    WorldState,
)


class _FakeUser32:
    """Deterministic Win32 user32 mock verifying click-through and focus-free calls."""

    def __init__(self, initial_style: int = 0x00000100, root_hwnd: int = 0x9999) -> None:
        self.style_by_hwnd: dict[int, int] = {root_hwnd: initial_style}
        self.root_hwnd = root_hwnd
        self.owner_cleared: list[tuple[int, int, int]] = []
        self.layered_calls: list[tuple[int, int, int, int]] = []
        self.set_window_pos_calls: list[tuple[int, int, int, int, int, int, int]] = []
        self.show_window_calls: list[tuple[int, int]] = []
        self.display_affinity_calls: list[tuple[int, int]] = []

    def GetAncestor(self, hwnd: Any, flag: int) -> int:
        assert flag == GA_ROOT
        return self.root_hwnd

    def GetWindowLongW(self, hwnd: Any, idx: int) -> int:
        assert idx == GWL_EXSTYLE
        return self.style_by_hwnd.get(int(hwnd.value), 0)

    def SetWindowLongW(self, hwnd: Any, idx: int, val: int) -> int:
        if idx == GWL_EXSTYLE:
            prev = self.style_by_hwnd.get(int(hwnd.value), 0)
            self.style_by_hwnd[int(hwnd.value)] = int(val)
            return prev
        self.owner_cleared.append((int(hwnd.value), idx, int(val)))
        return 0

    def SetWindowLongPtrW(self, hwnd: Any, idx: int, val: int) -> int:
        self.owner_cleared.append((int(hwnd.value), idx, int(val)))
        return 0

    def SetLayeredWindowAttributes(self, hwnd: Any, cr_key: Any, alpha: Any, flags: Any) -> int:
        self.layered_calls.append((int(hwnd.value), int(cr_key.value), int(alpha.value), int(flags.value)))
        return 1

    def SetWindowPos(self, hwnd: Any, insert_after: Any, x: int, y: int, cx: int, cy: int, flags: int) -> int:
        ins = int(insert_after.value) if hasattr(insert_after, "value") else int(insert_after)
        self.set_window_pos_calls.append((int(hwnd.value), ins, x, y, cx, cy, flags))
        return 1

    def ShowWindow(self, hwnd: Any, cmd: int) -> int:
        self.show_window_calls.append((int(hwnd.value), int(cmd)))
        return 1

    def SetWindowDisplayAffinity(self, hwnd: Any, affinity: Any) -> int:
        self.display_affinity_calls.append((int(hwnd.value), int(affinity.value)))
        return 1


class _FakeCanvas:
    """Captures Tkinter canvas primitives for assertion."""

    def __init__(self) -> None:
        self.items: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.deleted: int = 0

    def delete(self, tag: str) -> None:
        self.deleted += 1
        self.items.clear()

    def create_rectangle(self, *args: Any, **kwargs: Any) -> int:
        self.items.append(("rect", args, kwargs))
        return len(self.items)

    def create_oval(self, *args: Any, **kwargs: Any) -> int:
        self.items.append(("oval", args, kwargs))
        return len(self.items)

    def create_line(self, *args: Any, **kwargs: Any) -> int:
        self.items.append(("line", args, kwargs))
        return len(self.items)

    def create_text(self, *args: Any, **kwargs: Any) -> int:
        self.items.append(("text", args, kwargs))
        return len(self.items)


def _sample_world() -> WorldState:
    snap = PerceptionSnapshot(
        timestamp=1234.5,
        frame_width=1920,
        frame_height=1080,
        hp_ratio=0.91,
        mp_ratio=0.77,
        coords=None,
        minimap=MinimapReading(),
        detector_backend="HUD+UI",
        timings_ms={"total": 4.2},
    )
    return WorldState(
        snapshot=snap,
        coords_smoothed=Point2D(1075.0, 2631.0),
        camera_heading_deg=90.0,
        heading_source="minimap",
        is_stuck=False,
        consecutive_stuck=0,
        zero_hp_streak=0,
        coords_age_sec=0.0,
        waypoint_index=0,
        last_action_summary="walk->merchant",
    )


def test_win32_clickthrough_styles_noactivate_and_capture_exclusion() -> None:
    u32 = _FakeUser32(initial_style=0x00000100, root_hwnd=0x4242)
    info = apply_clickthrough_styles(
        0x1234,
        color_key_bgr=TRANSPARENT_COLOR_BGR,
        exclude_from_capture=True,
        detach_owner=True,
        user32_override=u32,
    )

    assert info["applied"] is True
    assert info["hwnd"] == 0x4242
    assert info["capture_excluded"] is True
    ex_style = int(info["ex_style"])
    assert (ex_style & WS_EX_LAYERED) == WS_EX_LAYERED
    assert (ex_style & WS_EX_TRANSPARENT) == WS_EX_TRANSPARENT
    assert (ex_style & WS_EX_TOPMOST) == WS_EX_TOPMOST
    assert (ex_style & WS_EX_NOACTIVATE) == WS_EX_NOACTIVATE
    assert (ex_style & CLICKTHROUGH_EXSTYLE_MASK) == CLICKTHROUGH_EXSTYLE_MASK

    assert u32.owner_cleared == [(0x4242, GWLP_HWNDPARENT, 0)]
    assert u32.layered_calls == [(0x4242, TRANSPARENT_COLOR_BGR, 255, LWA_COLORKEY)]
    assert u32.display_affinity_calls == [(0x4242, WDA_EXCLUDEFROMCAPTURE)]

    assert position_overlay_window(0x1234, 100, 50, 1920, 1080, visible=True, user32_override=u32) is True
    last_pos = u32.set_window_pos_calls[-1]
    assert last_pos[0] == 0x4242
    assert last_pos[1] == int(wintypes.HWND(HWND_TOPMOST).value)
    assert last_pos[2:6] == (100, 50, 1920, 1080)
    assert (last_pos[6] & (SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_SHOWWINDOW)) == (
        SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_SHOWWINDOW
    )
    assert u32.show_window_calls[-1] == (0x4242, SW_SHOWNOACTIVATE)

    assert position_overlay_window(0x1234, 0, 0, 1, 1, visible=False, user32_override=u32) is True
    assert u32.show_window_calls[-1] == (0x4242, SW_HIDE)


def test_build_overlay_frame_data_and_draw_banner_and_self_layers() -> None:
    cfg = load_config("config/default.yaml")
    world = _sample_world()
    data = build_overlay_frame_data(
        game_rect=(0, 0, 1920, 1080),
        world=world,
        fsm_state=BotState.REPAIR_AT_NPC,
        config=cfg,
        loop_hz=28.5,
        vision_ms=4.2,
        action_summary="walk->merchant",
        backend="HUD+UI",
        travel_desc="REPAIR:WALK_TO_MERCHANT",
        deaths_count=1,
        repairs_count=2,
        repair_rem_min=75.0,
    )

    assert data.phase == "REPAIR_AT_NPC"
    assert data.action == "walk->merchant"
    assert data.loop_hz == 28.5
    assert data.vision_ms == 4.2
    assert data.coords == (1075, 2631)
    assert data.self_char_box is not None and data.self_core_box is not None

    canvas = _FakeCanvas()
    drawn = draw_overlay_scene(canvas, data)
    assert drawn >= 8
    assert canvas.deleted == 1

    rect_outlines = [kw.get("outline") for kind, _a, kw in canvas.items if kind == "rect"]
    texts = [str(kw.get("text", "")) for kind, _a, kw in canvas.items if kind == "text"]

    assert "#64748B" in rect_outlines
    assert "#38BDF8" in rect_outlines

    joined = " | ".join(texts)
    assert "PHASE: REPAIR_AT_NPC (REPAIR:WALK_TO_MERCHANT)" in joined
    assert "walk->merchant" in joined
    assert "HZ: 28.5" in joined
    assert "VIS: 4.2ms" in joined
    assert "POS: (1075,2631)" in joined
    assert "D:1 R:2" in joined
    assert "[F7]:Rep [F6]:OV" in joined
    assert "SELF" in joined


def test_self_exclusion_boxes_scale_for_1080p_and_1024x768() -> None:
    c1, s1, core1 = compute_self_exclusion_boxes(1920, 1080, (0.50, 0.52), hud_fixed_pixel_ui=False)
    assert c1 == (960, 562)
    assert s1[0] < core1[0] < c1[0] < core1[2] < s1[2]
    assert s1[1] < core1[1] < c1[1] < core1[3] < s1[3]

    c2, s2, core2 = compute_self_exclusion_boxes(1024, 768, (0.50, 0.52), hud_fixed_pixel_ui=True)
    assert c2 == (512, 399)
    assert (s2[2] - s2[0]) >= 90
    assert s2[0] < core2[0] < c2[0] < core2[2] < s2[2]


def test_safety_monitor_f6_overlay_hotkey_and_engine_toggle_and_cli_flag() -> None:
    cfg = load_config("config/default.yaml")
    assert str(cfg.safety.get("overlay_toggle", "")).upper() == "F6"
    assert str(cfg.safety.get("dataset_capture_key", "")).upper() == "F8"
    assert str(cfg.safety.get("dataset_raw_dir", "")) == "dataset/raw_images"

    parser = build_parser()
    ns_default = parser.parse_args(["run", "--dry-run"])
    assert ns_default.debug_overlay is False
    ns_ov = parser.parse_args(["run", "--dry-run", "--debug-overlay"])
    assert ns_ov.debug_overlay is True

    safety = SafetyMonitor(cfg, enable_hotkeys=False)

    class _FakeHotkeyPoller:
        def __init__(self) -> None:
            self.calls = 0

        def poll(self) -> set[str]:
            self.calls += 1
            return {"overlay"} if self.calls == 1 else set()

    safety.hotkeys = _FakeHotkeyPoller()  # type: ignore[assignment]
    res1 = safety.check(None, is_running_active=False)
    assert res1.toggle_overlay is True
    assert res1.kill_triggered is False
    res2 = safety.check(None, is_running_active=False)
    assert res2.toggle_overlay is False

    toggled_states: list[bool] = []

    class _StubOverlay(TransparentDebugOverlay):
        def start(self) -> None:
            self._started = True

        def stop(self) -> None:
            self._started = False

    ov = _StubOverlay(enabled=False, on_toggle=lambda st: toggled_states.append(st))
    engine = BotEngine(
        config=cfg,
        mode=BotMode.FULL,
        live_input=False,
        show_overlay=False,
        debug_overlay=False,
        transparent_overlay=ov,
    )
    try:
        assert ov.is_enabled is False
        engine._handle_overlay_toggle(safety)
        assert ov.is_enabled is True
        assert engine.debug_overlay_enabled is True
        assert toggled_states == [True]

        engine._handle_overlay_toggle(safety)
        assert ov.is_enabled is False
        assert engine.debug_overlay_enabled is False
        assert toggled_states == [True, False]
    finally:
        engine.pipeline.close()


def test_transparent_overlay_real_tk_attached_lifecycle_and_clickthrough() -> None:
    import tkinter as tk

    cfg = load_config("config/default.yaml")
    world = _sample_world()
    root = tk.Tk()
    root.withdraw()
    ov = TransparentDebugOverlay(master=root, enabled=True, refresh_ms=25)
    try:
        ov.start()
        assert ov.style_info.get("applied") is True
        ex_style = int(ov.style_info.get("ex_style", 0))
        assert (ex_style & WS_EX_LAYERED) == WS_EX_LAYERED
        assert (ex_style & WS_EX_TRANSPARENT) == WS_EX_TRANSPARENT
        assert (ex_style & WS_EX_TOPMOST) == WS_EX_TOPMOST
        assert (ex_style & WS_EX_NOACTIVATE) == WS_EX_NOACTIVATE

        data = build_overlay_frame_data(
            game_rect=(0, 0, 1024, 768),
            world=world,
            fsm_state=BotState.AT_SPOT_IDLE,
            config=cfg,
            loop_hz=25.0,
            vision_ms=4.2,
            action_summary="idle_at_spot",
        )
        ov.update(data)
        ov._on_timer_tick()
        root.update_idletasks()
        assert ov._is_shown is True
        assert ov._canvas is not None
        assert len(ov._canvas.find_all()) >= 8

        assert ov.toggle() is False
        ov._on_timer_tick()
        root.update_idletasks()
        assert ov._is_shown is False
    finally:
        ov.stop()
        try:
            root.update_idletasks()
        except Exception:
            pass
        root.destroy()


def test_f8_clean_dataset_capture_hotkey_and_saved_frame_notice(tmp_path) -> None:
    """Verify F8 hotkey sets dataset_capture_requested and saves clean raw 1024x768 JPEG 98 with SAVED_FRAME(N)."""
    import cv2
    import numpy as np
    from vikings_bot.capture.screen import FramePacket

    cfg = load_config("config/default.yaml")
    safety = SafetyMonitor(cfg, enable_hotkeys=False)

    class _FakeF8Poller:
        def poll(self) -> set[str]:
            return {"dataset_capture"}

    safety.hotkeys = _FakeF8Poller()  # type: ignore[assignment]
    chk = safety.check(None, is_running_active=False)
    assert chk.dataset_capture_requested is True

    engine = BotEngine(cfg, mode=BotMode.FULL, live_input=False)
    raw_img = np.full((768, 1024, 3), (40, 95, 55), dtype=np.uint8)
    pkt = FramePacket(image_bgr=raw_img, timestamp=1.0, source_name="vm_1024_clean")

    p1, cnt1 = engine.save_clean_dataset_frame(pkt, out_dir_override=tmp_path)
    assert p1 is not None and p1.exists() and p1.name.startswith("ryl_1024_") and p1.suffix == ".jpg"
    assert cnt1 == 1 and engine._last_saved_notice == "SAVED_FRAME (1)"
    loaded = cv2.imread(str(p1), cv2.IMREAD_COLOR)
    assert loaded is not None and loaded.shape == (768, 1024, 3)

    p2, cnt2 = engine.save_clean_dataset_frame(pkt, out_dir_override=tmp_path)
    assert p2 is not None and p2.exists() and p2 != p1
    assert cnt2 == 2 and engine._last_saved_notice == "SAVED_FRAME (2)"
