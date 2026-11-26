"""Command-line interface for Vikings Bot v2 (Mouse Mode • VM-Ready • AHK v2 • Pure OpenCV)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from vikings_bot.capture.screen import FramePacket, PrintWindowCaptureSource
from vikings_bot.config import BotConfig, load_config
from vikings_bot.input.ahk_driver import AhkInputDriver, find_ahk_executable
from vikings_bot.perception.coords import calibrate_digit_templates
from vikings_bot.runtime.engine import BotEngine
from vikings_bot.state.models import BotMode

DEFAULT_CALIBRATION_SAMPLES: list[tuple[str, str]] = [
    ("Vikings Online_Mortega-_2026_11_22_04_06(1).jpg", "617,1822"),
    ("Vikings Online_Mortega-_2026_11_22_04_06(6).jpg", "601,1846"),
    ("Vikings Online_Mortega-_2026_11_22_04_06(11).jpg", "599,1846"),
    ("Vikings Online_Mortega-_2026_11_22_04_07(15).jpg", "603,1844"),
    ("Vikings Online_Mortega-_2026_11_22_04_07(16).jpg", "603,1844"),
    ("Vikings Online_Mortega-_2026_11_22_04_07(20).jpg", "618,1836"),
    ("Vikings Online_Mortega-_2026_11_22_04_07(25).jpg", "618,1836"),
    ("Vikings Online_Mortega-_2026_11_22_04_09(30).jpg", "615,1835"),
    ("Vikings Online_Baddies_2026_10_30_06_49(1).jpg", "1104,1102"),
    ("Vikings Online_Baddies_2026_10_30_06_59(120).jpg", "1142,1540"),
    ("Vikings Online_Baddies_2026_10_30_07_05(236).jpg", "995,1125"),
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _apply_runtime_flags(cfg: BotConfig, args: argparse.Namespace) -> None:
    if getattr(args, "vm_1024", False):
        cfg.perception["hud_fixed_pixel_ui"] = True
        cfg.controls["mode"] = "mouse"
        cfg.controls["vm_absolute_mouse"] = True
        cfg.respawn_travel["mouse_only_travel"] = True
    if getattr(args, "bg_mode", False):
        cfg.controls["input_backend"] = "ahk"
        cfg.controls["ahk_mode"] = "host_background"
        cfg.window["capture_backend"] = "printwindow"
        cfg.window["require_focus"] = False


def _cmd_calibrate_digits(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    ss_dir = Path(args.screenshots)
    out_dir = cfg.resolve_path(cfg.perception.get("digit_templates_dir", "assets/digit_templates"))
    coord_roi = cfg.roi("coords")

    labeled_crops = []
    for fname, expected in DEFAULT_CALIBRATION_SAMPLES:
        fpath = ss_dir / fname
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pkt = FramePacket(image_bgr=img, timestamp=0.0, source_name=fname)
        crop, _ = pkt.crop_norm(coord_roi)
        labeled_crops.append((crop, expected))

    if not labeled_crops:
        print(f"No calibration screenshots found in {ss_dir}")
        return 1

    counts = calibrate_digit_templates(labeled_crops, out_dir)
    print(f"Calibrated {len(counts)} glyph templates into {out_dir}:")
    for ch in sorted(counts):
        print(f"  '{ch}': {counts[ch]} sample(s)")
    missing = set("0123456789,") - set(counts)
    if missing:
        print(f"WARNING: Missing glyphs: {sorted(missing)}")
        return 2
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, route_override=args.route)
    _apply_runtime_flags(cfg, args)
    engine = BotEngine(
        config=cfg,
        mode=BotMode(args.mode),
        live_input=False,
        show_overlay=not args.headless,
    )
    stats = engine.run_replay(
        screenshots_dir=args.screenshots,
        name_filter=args.filter,
        delay_ms=args.delay_ms,
        save_annotated_dir=args.save_dir,
        headless=args.headless,
    )
    print("Replay summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    import time
    from vikings_bot.capture.screen import create_capture_source
    from vikings_bot.capture.window import WindowTracker
    from vikings_bot.input.driver import Win32InputDriver
    from vikings_bot.perception.pipeline import PerceptionPipeline

    cfg = load_config(args.config)
    _apply_runtime_flags(cfg, args)
    is_admin = Win32InputDriver.is_admin()
    ahk_exe = find_ahk_executable(cfg.controls.get("ahk_exe_path"))
    print(f"[1] Run as Administrator : {'YES (OK)' if is_admin else 'NO  <-- WARNING: Run elevated for Vikings Online!'}")
    print(f"[2] AutoHotkey v2 Binary : {ahk_exe if ahk_exe else 'NOT FOUND (install AutoHotkey v2 or compile bridge .exe)'}")

    wt = WindowTracker(
        process_name=str(cfg.window.get("process_name", "Minia.exe")),
        title_substring=str(cfg.window.get("title_substring", "Vikings")),
        process_aliases=list(cfg.window.get("process_aliases") or []),
        title_aliases=list(cfg.window.get("title_aliases") or []),
    )
    win = wt.find_window()
    if win is not None:
        print(f"[3] Game Window Found    : YES (hwnd=0x{win.hwnd:X}, pid={win.pid}, rect={win.width}x{win.height} at ({win.left},{win.top}), focused={wt.is_focused(win)})")
    else:
        _fg, fg_pid, fg_title = wt.get_foreground_info()
        print(f"[3] Game Window Found    : NO  (foreground now: pid={fg_pid}, title={fg_title!r})")

    cap = create_capture_source(
        backend=str(cfg.window.get("capture_backend", "mss")),
        window_tracker=wt,
        target_fps=15.0,
    )
    cap.start()
    frame = None
    for _ in range(20):
        frame = cap.latest_frame()
        if frame is not None:
            break
        time.sleep(0.1)
    cap.stop()

    if frame is None:
        print("[4] Screen Capture       : FAILED (no frame received)")
        return 1

    luma = frame.mean_luminance
    luma_status = "OK" if luma >= 5.0 else "BLACK FRAME!"
    print(f"[4] Screen Capture       : {frame.width}x{frame.height} via {frame.source_name}, mean_luma={luma:.1f} -> {luma_status}")

    pipe = PerceptionPipeline(cfg)
    snap = pipe.analyze(frame)
    pos = f"({snap.coords.x}, {snap.coords.y}) conf={snap.coords.confidence:.2f}" if snap.coords else "None"
    print(
        f"[5] Live Gold Perception : HP={snap.hp_ratio:.0%} MP={snap.mp_ratio:.0%} | Coords={pos} | "
        f"GoldRustlers={len(snap.mobs)}(skip={len(snap.contested_mobs)}) | GoldBags={len(snap.loot)} | "
        f"Mode={snap.interface_mode} | {snap.timings_ms['total']:.1f}ms"
    )
    return 0


def _cmd_test_ahk_bg(args: argparse.Namespace) -> int:
    """Test whether PrintWindow capture and AHK v2 ControlClick/ControlSend work on unfocused RYL."""
    from vikings_bot.capture.window import WindowTracker

    cfg = load_config(args.config)
    wt = WindowTracker(
        process_name=str(cfg.window.get("process_name", "Minia.exe")),
        title_substring=str(cfg.window.get("title_substring", "Vikings")),
        process_aliases=list(cfg.window.get("process_aliases") or []),
        title_aliases=list(cfg.window.get("title_aliases") or []),
    )
    win = wt.find_window()
    if win is None:
        print("[FAIL] Vikings Online window not found. Open RYL first before testing background mode.")
        return 1
    print(f"[1] Target RYL Window    : hwnd=0x{win.hwnd:X} ({win.width}x{win.height}) focused={wt.is_focused(win)}")

    bgr = PrintWindowCaptureSource.grab_hwnd_bgr(win.hwnd, win.width, win.height)
    if bgr is None:
        print("[2] PrintWindow Capture  : FAILED (could not grab DC)")
    else:
        luma = float(bgr[::16, ::16].mean())
        ok_str = "OK (non-black frame!)" if luma >= 5.0 else "BLACK FRAME (RYL DirectX blocked PrintWindow; use VMware vm_foreground!)"
        print(f"[2] PrintWindow Capture  : {bgr.shape[1]}x{bgr.shape[0]} mean_luma={luma:.1f} -> {ok_str}")

    ahk_exe = find_ahk_executable(cfg.controls.get("ahk_exe_path"))
    if ahk_exe is None:
        print("[3] AutoHotkey v2 Bridge : NOT INSTALLED (install AutoHotkey v2 from autohotkey.com)")
        return 2

    script_p = cfg.resolve_path(str(cfg.controls.get("ahk_script_path", "ahk/vikings_input_bridge.ahk")))
    drv = AhkInputDriver(
        ahk_mode="host_background",
        script_path=script_p,
        ahk_exe=str(ahk_exe),
        window_tracker=wt,
    )
    try:
        ok, reply = drv.health_check()
        print(f"[3] AutoHotkey v2 Bridge : {'OK (' + reply + ')' if ok else 'FAILED (' + reply + ')'}")
        if ok and args.send_probe:
            print("[4] Sending non-destructive ControlSend('e') probe to RYL background window...")
            drv.tap_key("e", hold_sec=0.04)
            print("[4] Probe sent. Verify in-game whether 'e' registered without focusing RYL.")
    finally:
        drv.close()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, route_override=args.route)
    _apply_runtime_flags(cfg, args)
    engine = BotEngine(
        config=cfg,
        mode=BotMode(args.mode),
        live_input=bool(args.live),
        show_overlay=args.overlay,
        debug_overlay=bool(getattr(args, "debug_overlay", False)),
        auto_start=args.auto_start,
    )
    engine.run_live()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vikings-bot",
        description="Vikings Bot v2 — Mouse Mode • VM-Ready • AHK v2 • Pure OpenCV (Gold Rustler)",
    )
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    common_sub = argparse.ArgumentParser(add_help=False)
    common_sub.add_argument("--config", default=argparse.SUPPRESS, help="Path to YAML config file")
    common_sub.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser(
        "calibrate-digits", parents=[common_sub], help="Extract bitmap digit templates (0-9, ',') from screenshots"
    )
    p_cal.add_argument(
        "--screenshots",
        default=r"C:\Program Files (x86)\Vikings Online\ScreenShots",
        help="Path to Vikings Online ScreenShots folder",
    )
    p_cal.set_defaults(func=_cmd_calibrate_digits)

    p_rep = sub.add_parser(
        "replay", parents=[common_sub], help="Replay offline screenshots through Pure OpenCV Perception + FSM + Overlay"
    )
    p_rep.add_argument(
        "--screenshots",
        default=r"C:\Program Files (x86)\Vikings Online\ScreenShots",
        help="Path to screenshots folder",
    )
    p_rep.add_argument("--filter", default="Mortega-", help="Substring filter for screenshot filenames")
    p_rep.add_argument("--mode", default="full", choices=[m.value for m in BotMode], help="FSM mode")
    p_rep.add_argument("--route", default=None, help="Override waypoint route name")
    p_rep.add_argument("--target", default="gold", choices=["gold"], help="Target farming profile (default: gold)")
    p_rep.add_argument("--vm-1024", action="store_true", help="Enable VMware 1024x768 fixed-pixel UI preset")
    p_rep.add_argument("--delay-ms", type=int, default=350, help="Frame display delay in ms")
    p_rep.add_argument("--save-dir", default=None, help="Optional folder to save annotated overlay JPGs")
    p_rep.add_argument("--headless", action="store_true", help="Run without opening OpenCV window")
    p_rep.set_defaults(func=_cmd_replay)

    p_doc = sub.add_parser(
        "doctor", parents=[common_sub], help="Diagnose Admin/UIPI, AHK v2, game window, capture, and live Gold perception"
    )
    p_doc.add_argument("--vm-1024", action="store_true", help="Use VMware 1024x768 fixed-pixel UI preset")
    p_doc.add_argument("--bg-mode", action="store_true", help="Test PrintWindow + AHK host_background preset")
    p_doc.set_defaults(func=_cmd_doctor)

    p_bg = sub.add_parser(
        "test-ahk-bg", parents=[common_sub], help="Test Win32 PrintWindow + AHK v2 ControlClick/ControlSend in background"
    )
    p_bg.add_argument("--send-probe", action="store_true", help="Send a single non-destructive 'e' ControlSend probe")
    p_bg.set_defaults(func=_cmd_test_ahk_bg)

    p_run = sub.add_parser("run", parents=[common_sub], help="Run live Mouse-Mode farming loop against Vikings Online")
    mode_grp = p_run.add_mutually_exclusive_group()
    mode_grp.add_argument("--dry-run", action="store_true", help="Capture & log only (default when neither flag is set)")
    mode_grp.add_argument("--live", action="store_true", help="Enable AHK v2 live input control")
    p_run.add_argument("--mode", default="full", choices=[m.value for m in BotMode], help="Bot mode")
    p_run.add_argument("--target", default="gold", choices=["gold"], help="Target farming profile (default: gold)")
    p_run.add_argument("--route", default=None, help="Override waypoint route name")
    p_run.add_argument("--vm-1024", action="store_true", help="Enable VMware 1024x768 fixed-pixel UI + Pure Mouse preset")
    p_run.add_argument("--bg-mode", action="store_true", help="Use experimental PrintWindow + AHK host_background mode")
    p_run.add_argument("--overlay", action="store_true", help="Show OpenCV overlay window")
    p_run.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Show Win32 transparent click-through debug overlay over the RYL window (toggle with F6)",
    )
    p_run.add_argument("--auto-start", action="store_true", help="Start active immediately without waiting for F9")
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
