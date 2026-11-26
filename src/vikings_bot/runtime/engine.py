"""Main fixed-rate orchestrator tying capture, Pure OpenCV Gold perception, Mouse-Mode FSM, AHK v2 driver, safety, and overlay."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from vikings_bot.brain.fsm import FarmingFSM
from vikings_bot.capture.screen import FrameSource, MssCaptureSource, ReplayFrameSource, create_capture_source
from vikings_bot.capture.window import WindowTracker
from vikings_bot.config import BotConfig
from vikings_bot.debug_overlay import TransparentDebugOverlay, build_overlay_frame_data
from vikings_bot.input.actions import ActionExecutor
from vikings_bot.input.driver import Win32InputDriver, create_input_driver
from vikings_bot.perception.pipeline import PerceptionPipeline
from vikings_bot.runtime.overlay import DebugOverlay
from vikings_bot.runtime.recorder import SnapshotRecorder
from vikings_bot.runtime.safety import SafetyMonitor
from vikings_bot.runtime.tracer import SessionTracer
from vikings_bot.state.models import BotMode, BotState
from vikings_bot.state.tracker import WorldStateTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineTelemetry:
    """Thread-safe snapshot of live bot state."""

    state: str
    target_id: str
    control_mode: str
    live_input: bool
    is_admin: bool
    window_found: bool
    window_focused: bool
    window_desc: str
    coords: tuple[int, int] | None
    hp_ratio: float
    mp_ratio: float
    mobs_count: int
    loot_count: int
    kills_count: int
    fps: float
    uptime_sec: float
    last_action: str
    deaths_count: int = 0
    travel_desc: str = ""
    detector_backend: str = "OpenCV-Gold"
    repairs_count: int = 0
    repair_rem_min: float | None = None


class BotEngine:
    """Runs live (--dry-run / --live) or offline replay loops with AHK v2 input and click-through overlay."""

    def __init__(
        self,
        config: BotConfig,
        mode: BotMode = BotMode.FULL,
        live_input: bool = False,
        show_overlay: bool = False,
        auto_start: bool = False,
        target_id: str = "gold",
        on_status: Callable[[EngineTelemetry], None] | None = None,
        debug_overlay: bool = False,
        transparent_overlay: TransparentDebugOverlay | None = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.live_input = live_input
        self.show_overlay = show_overlay
        self.debug_overlay_enabled = bool(debug_overlay)
        self._transparent_overlay: TransparentDebugOverlay | None = transparent_overlay
        self._owns_transparent_overlay: bool = False
        self.auto_start = auto_start
        self.target_id = target_id
        self.on_status = on_status
        self._stop_event = threading.Event()
        self._toggle_pause_event = threading.Event()
        self._toggle_overlay_event = threading.Event()

        self.window_tracker = WindowTracker(
            process_name=str(config.window.get("process_name", "Minia.exe")),
            title_substring=str(config.window.get("title_substring", "Vikings")),
            process_aliases=list(config.window.get("process_aliases") or []),
            title_aliases=list(config.window.get("title_aliases") or []),
        )
        self.pipeline = PerceptionPipeline(config)
        self.tracker = WorldStateTracker(
            coord_max_jump=float(config.perception.get("coord_max_jump", 120.0)),
            stuck_timeout_sec=float(config.farming.get("stuck_timeout_sec", 4.0)),
            stuck_min_move_units=float(config.farming.get("stuck_min_move_units", 2.5)),
            death_zero_hp_frames=int(config.safety.get("death_zero_hp_frames", 12)),
            lock_iou_threshold=float(config.perception.get("lock_iou_threshold", 0.30)),
            lock_dist_threshold_px=float(config.perception.get("lock_dist_threshold_px", 85.0)),
        )
        self.tracker.last_kill_radius_px = float(config.perception.get("last_kill_loot_radius_px", 140.0))
        self.fsm = FarmingFSM(config, self.tracker, mode=mode)
        self.driver = create_input_driver(config, live_input=live_input, window_tracker=self.window_tracker)
        self.executor = ActionExecutor(
            driver=self.driver,
            tracker=self.tracker,
            click_jitter_px=int(config.controls.get("click_jitter_px", 0)),
        )
        self.overlay = DebugOverlay(config)
        self.recorder = SnapshotRecorder(config)
        self.tracer = SessionTracer(
            trace_path=config.resolve_path("logs/latest_trace.jsonl"),
            frames_dir=config.resolve_path("logs/trace_frames"),
        )

    def request_stop(self) -> None:
        self._stop_event.set()

    def request_toggle_pause(self) -> None:
        self._toggle_pause_event.set()

    def request_repair_now(self) -> None:
        self.fsm.request_repair_now()

    def request_toggle_overlay(self) -> None:
        self._toggle_overlay_event.set()

    def set_debug_overlay_enabled(self, enabled: bool) -> None:
        self.debug_overlay_enabled = bool(enabled)
        if self._transparent_overlay is not None:
            self._transparent_overlay.set_enabled(self.debug_overlay_enabled, notify=False)

    def _ensure_transparent_overlay(self, enabled: bool) -> TransparentDebugOverlay | None:
        if self._transparent_overlay is None:
            self._transparent_overlay = TransparentDebugOverlay(enabled=enabled)
            self._owns_transparent_overlay = True
        self._transparent_overlay.start()
        self._transparent_overlay.set_enabled(enabled, notify=False)
        return self._transparent_overlay

    def _handle_overlay_toggle(self, safety: SafetyMonitor) -> None:
        if self._transparent_overlay is None:
            self._ensure_transparent_overlay(enabled=True)
            self.debug_overlay_enabled = True
            safety.audio.play("overlay_on")
            return
        self._transparent_overlay.start()
        new_state = self._transparent_overlay.toggle()
        self.debug_overlay_enabled = new_state
        safety.audio.play("overlay_on" if new_state else "overlay_off")

    def run_live(self) -> None:
        """Run real-time loop against the Vikings Online window (VM Foreground or Host Background)."""
        wcfg = self.config.window
        ahk_mode = str(self.config.controls.get("ahk_mode", "vm_foreground")).lower()
        is_bg = ahk_mode == "host_background"
        cap_backend = str(wcfg.get("capture_backend", "printwindow" if is_bg else "mss"))

        capture: FrameSource = create_capture_source(
            backend=cap_backend,
            window_tracker=self.window_tracker,
            target_fps=float(wcfg.get("capture_fps", 25)),
            black_frame_threshold=float(wcfg.get("black_frame_threshold", 2.0)),
            black_frame_limit=int(wcfg.get("black_frame_limit", 15)),
        )
        safety = SafetyMonitor(self.config, window_tracker=self.window_tracker, enable_hotkeys=True)
        if is_bg:
            safety.require_focus = False

        tick_period = 1.0 / max(1.0, float(self.config.farming.get("tick_hz", 20.0)))
        is_admin = Win32InputDriver.is_admin()
        in_label = "AHK-BG" if is_bg else ("AHK-VM" if str(self.config.controls.get("input_backend", "ahk")).lower() == "ahk" else "WIN32")

        logger.info(
            "Starting Vikings Bot v2 (%s | input=%s | cap=%s | admin=%s | route=%s). "
            "Hotkeys: %s=Pause/Resume, %s=Kill Switch, %s=Overlay.",
            "LIVE INPUT" if self.live_input else "DRY-RUN",
            in_label,
            cap_backend,
            "YES" if is_admin else "NO",
            self.config.active_route.name,
            self.config.safety.get("pause_toggle", "F9"),
            self.config.safety.get("kill_switch", "F10"),
            self.config.safety.get("overlay_toggle", "F6"),
        )

        if self._transparent_overlay is not None:
            self._transparent_overlay.set_enabled(self.debug_overlay_enabled, notify=False)
        elif self.debug_overlay_enabled:
            self._ensure_transparent_overlay(enabled=True)

        capture.start()
        if self.auto_start:
            self.fsm.resume(time.monotonic())
            safety.audio.play("resume")

        tick_count = 0
        fps_t0 = time.perf_counter()
        session_t0 = time.monotonic()
        fps_val = 0.0
        last_heartbeat_ts = 0.0
        last_ui_ts = 0.0
        last_imode_check_ts = -999.0
        black_snapshot_saved = False
        prev_state = self.fsm.state

        try:
            while self.fsm.state != BotState.EMERGENCY_STOP:
                loop_t0 = time.perf_counter()
                now = time.monotonic()

                if self._stop_event.is_set():
                    self.executor.release_all()
                    self.fsm.emergency_stop(now, "gui_stop")
                    break

                if self.fsm.state == BotState.PAUSED:
                    chk_p = safety.check(None, is_running_active=False)
                    if chk_p.kill_triggered or self._stop_event.is_set():
                        self.executor.release_all()
                        self.fsm.emergency_stop(now, "hotkey_kill" if chk_p.kill_triggered else "gui_stop")
                        break
                    gui_ov_p = self._toggle_overlay_event.is_set()
                    if gui_ov_p:
                        self._toggle_overlay_event.clear()
                    if chk_p.toggle_overlay or gui_ov_p:
                        self._handle_overlay_toggle(safety)
                    gui_toggle_p = self._toggle_pause_event.is_set()
                    if gui_toggle_p:
                        self._toggle_pause_event.clear()
                    if chk_p.toggle_pause or gui_toggle_p:
                        self.fsm.resume(now)
                        safety.audio.play("resume")
                    else:
                        ov_active_p = (
                            self._transparent_overlay is not None and self._transparent_overlay.is_enabled
                        )
                        if (self.on_status is not None or ov_active_p) and (now - last_ui_ts >= 0.25):
                            last_ui_ts = now
                            frame_p = capture.latest_frame()
                            pos_tuple_p = None
                            hp_p, mp_p = 1.0, 1.0
                            if frame_p is not None:
                                snap_p = self.pipeline.analyze(
                                    frame_p, check_tele_ui=False, skip_field_detectors=True
                                )
                                hp_p, mp_p = snap_p.hp_ratio, snap_p.mp_ratio
                                if snap_p.coords is not None:
                                    pos_tuple_p = (int(round(snap_p.coords.point.x)), int(round(snap_p.coords.point.y)))
                            win_p = self.window_tracker.find_window()
                            focused_p = self.window_tracker.is_focused(win_p)
                            if ov_active_p and self._transparent_overlay is not None:
                                if win_p is not None:
                                    self._transparent_overlay.update(
                                        build_overlay_frame_data(
                                            game_rect=(win_p.left, win_p.top, win_p.width, win_p.height),
                                            world=None,
                                            fsm_state=self.fsm.state,
                                            config=self.config,
                                            loop_hz=0.0,
                                            vision_ms=0.0,
                                            action_summary="PAUSED (press F9)",
                                            backend=self.pipeline.active_backend_desc,
                                            input_mode=in_label,
                                            game_focused=focused_p,
                                            deaths_count=self.fsm.deaths_count,
                                            repairs_count=self.fsm.repairs_count,
                                            repair_rem_min=self.fsm.repair_remaining_minutes,
                                            hp_ratio_override=hp_p,
                                            mp_ratio_override=mp_p,
                                            coords_override=pos_tuple_p,
                                        )
                                    )
                                else:
                                    self._transparent_overlay.clear_and_hide()
                            if self.on_status is not None:
                                win_desc_p = (
                                    f"{win_p.title or 'pid:' + str(win_p.pid)}({win_p.width}x{win_p.height})"
                                    if win_p
                                    else "NOT_FOUND"
                                )
                                try:
                                    self.on_status(
                                        EngineTelemetry(
                                            state=self.fsm.state.value,
                                            target_id=self.target_id,
                                            control_mode=self.fsm.control_mode,
                                            live_input=self.live_input,
                                            is_admin=is_admin,
                                            window_found=win_p is not None,
                                            window_focused=focused_p,
                                            window_desc=win_desc_p,
                                            coords=pos_tuple_p,
                                            hp_ratio=hp_p,
                                            mp_ratio=mp_p,
                                            mobs_count=0,
                                            loot_count=0,
                                            kills_count=self.fsm.kills_count,
                                            fps=0.0,
                                            uptime_sec=max(0.0, now - session_t0),
                                            last_action="PAUSED (press F9)",
                                            deaths_count=self.fsm.deaths_count,
                                            travel_desc="",
                                            detector_backend=self.pipeline.active_backend_desc,
                                            repairs_count=self.fsm.repairs_count,
                                            repair_rem_min=self.fsm.repair_remaining_minutes,
                                        )
                                    )
                                except Exception:
                                    pass
                        time.sleep(0.018)
                        continue

                frame = capture.latest_frame()
                if frame is None:
                    chk = safety.check(None, is_running_active=False)
                    if chk.kill_triggered:
                        self.fsm.emergency_stop(now, "hotkey_kill")
                        break
                    time.sleep(0.02)
                    continue

                frame = type(frame)(
                    image_bgr=frame.image_bgr,
                    timestamp=now,
                    origin_x=frame.origin_x,
                    origin_y=frame.origin_y,
                    source_name=frame.source_name,
                )
                need_tele_ui = self.fsm.state == BotState.TELEPORT_TO_SPOT
                need_repair_ui = self.fsm.state == BotState.REPAIR_AT_NPC
                skip_field = self.fsm.state in (
                    BotState.ENSURE_MOUSE_MODE,
                    BotState.DEAD_RESPAWN,
                    BotState.REPAIR_RECALL,
                    BotState.REPAIR_AT_NPC,
                    BotState.TELEPORT_TO_SPOT,
                    BotState.TRAVEL_TO_SPOT,
                )
                skip_mob_det = self.fsm.state in (BotState.COMBAT_WAIT, BotState.LOOT, BotState.ENSURE_MOUSE_MODE)
                loot_rad_lim = (
                    float(self.config.farming.get("loot_entry_close_dist_px", 220.0))
                    if self.fsm.state in (BotState.LOOT, BotState.COMBAT_WAIT)
                    else None
                )
                cursor_rel_xy: tuple[int, int] | None = None
                # One-shot interface mode check ONLY during ENSURE_MOUSE_MODE (never during active farming!)
                need_imode_check = self.fsm.state == BotState.ENSURE_MOUSE_MODE
                if need_imode_check:
                    last_imode_check_ts = now
                    cpos = Win32InputDriver.get_cursor_pos()
                    if cpos is not None:
                        cursor_rel_xy = (cpos[0] - frame.origin_x, cpos[1] - frame.origin_y)

                snapshot = self.pipeline.analyze(
                    frame,
                    check_tele_ui=need_tele_ui,
                    skip_field_detectors=skip_field,
                    cursor_rel_xy=cursor_rel_xy,
                    check_interface_mode=need_imode_check,
                    check_repair_ui=need_repair_ui,
                    skip_mob_detection=skip_mob_det,
                    loot_radius_limit_px=loot_rad_lim,
                )
                world = self.tracker.update(snapshot)

                active = self.fsm.state not in (BotState.PAUSED, BotState.EMERGENCY_STOP)
                in_rt = self.fsm.state in (
                    BotState.DEAD_RESPAWN,
                    BotState.REPAIR_RECALL,
                    BotState.REPAIR_AT_NPC,
                    BotState.TELEPORT_TO_SPOT,
                    BotState.TRAVEL_TO_SPOT,
                )
                chk = safety.check(world, is_running_active=active, in_respawn_or_travel=in_rt)

                if chk.kill_triggered or self._stop_event.is_set():
                    self.executor.release_all()
                    self.fsm.emergency_stop(now, "hotkey_kill" if chk.kill_triggered else "gui_stop")
                    break

                gui_ov = self._toggle_overlay_event.is_set()
                if gui_ov:
                    self._toggle_overlay_event.clear()
                if chk.toggle_overlay or gui_ov:
                    self._handle_overlay_toggle(safety)

                if chk.force_pause_reason is not None and active:
                    self.executor.release_all()
                    self.fsm.pause(now, chk.force_pause_reason)
                    logger.warning("Auto-paused: %s", chk.force_pause_reason)

                gui_toggle = self._toggle_pause_event.is_set()
                if gui_toggle:
                    self._toggle_pause_event.clear()

                if chk.toggle_pause or gui_toggle:
                    if self.fsm.state == BotState.PAUSED:
                        self.fsm.resume(now)
                        safety.audio.play("resume")
                    else:
                        self.executor.release_all()
                        self.fsm.pause(now, "hotkey_pause" if chk.toggle_pause else "gui_pause")
                        safety.audio.play("pause")

                need_ui = self.on_status is not None and (now - last_ui_ts >= 0.18)
                need_hb = (now - last_heartbeat_ts) >= 2.0
                if need_ui or need_hb:
                    win = self.window_tracker.find_window()
                    focused = self.window_tracker.is_focused(win)
                    win_desc = f"{win.title or 'pid:'+str(win.pid)}({win.width}x{win.height})" if win else "NOT_FOUND"
                    pos_tuple = (
                        (int(round(world.coords_smoothed.x)), int(round(world.coords_smoothed.y)))
                        if world.coords_smoothed is not None
                        else None
                    )
                    pos_desc = f"({pos_tuple[0]},{pos_tuple[1]})" if pos_tuple is not None else "None"

                    if need_ui and self.on_status is not None:
                        last_ui_ts = now
                        try:
                            self.on_status(
                                EngineTelemetry(
                                    state=self.fsm.state.value,
                                    target_id=self.target_id,
                                    control_mode=self.fsm.control_mode,
                                    live_input=self.live_input,
                                    is_admin=is_admin,
                                    window_found=win is not None,
                                    window_focused=focused,
                                    window_desc=win_desc,
                                    coords=pos_tuple,
                                    hp_ratio=world.hp_ratio,
                                    mp_ratio=world.mp_ratio,
                                    mobs_count=len(world.mobs),
                                    loot_count=len(world.loot),
                                    kills_count=self.fsm.kills_count,
                                    fps=fps_val,
                                    uptime_sec=max(0.0, now - session_t0),
                                    last_action=world.last_action_summary,
                                    deaths_count=self.fsm.deaths_count,
                                    travel_desc=self.fsm.travel_status_desc,
                                    detector_backend=self.pipeline.active_backend_desc,
                                    repairs_count=self.fsm.repairs_count,
                                    repair_rem_min=self.fsm.repair_remaining_minutes,
                                )
                            )
                        except Exception:
                            pass

                    if need_hb:
                        last_heartbeat_ts = now
                        rt_tag = f" ({self.fsm.travel_status_desc})" if self.fsm.travel_status_desc else ""
                        logger.info(
                            "STATUS | state=%s%s | mode=%s | win=%s focused=%s | pos=%s | hp=%.0f%% mp=%.0f%% | mobs=%d gold=%d kills=%d deaths=%d repairs=%d | vis=%.1fms fps=%.1f",
                            self.fsm.state.value,
                            rt_tag,
                            in_label,
                            win_desc,
                            focused,
                            pos_desc,
                            world.hp_ratio * 100.0,
                            world.mp_ratio * 100.0,
                            len(world.mobs),
                            len(world.loot),
                            self.fsm.kills_count,
                            self.fsm.deaths_count,
                            self.fsm.repairs_count,
                            float(snapshot.timings_ms.get("total", 0.0)),
                            fps_val,
                        )

                if isinstance(capture, MssCaptureSource) and capture.black_frame_warning and not black_snapshot_saved:
                    self.recorder.save_async("black_frame_warning", frame, world, self.fsm.state)
                    safety.audio.play("alert")
                    black_snapshot_saved = True

                intents = self.fsm.step(world)
                if self.fsm.last_blacklisted_loot is not None:
                    bx, by = self.fsm.last_blacklisted_loot
                    self.pipeline.loot.blacklist_point(bx, by, now, ttl_sec=120.0)

                exec_t0 = time.perf_counter()
                act_summary = ""
                if intents:
                    act_summary = self.executor.execute_many(
                        intents, now, origin_x=frame.origin_x, origin_y=frame.origin_y
                    )
                    if "drag_cam(" in act_summary:
                        self.fsm._post_drag_settle_until = time.monotonic() + 0.18
                exec_ms = (time.perf_counter() - exec_t0) * 1000.0
                active_cds = [k for k, exp in self.tracker._cooldowns.items() if exp > now]
                self.tracer.record_tick(
                    now=now,
                    state=self.fsm.state,
                    world=world,
                    intents_count=len(intents),
                    action_summary=act_summary,
                    analyze_ms=float(snapshot.timings_ms.get("total", 0.0)),
                    exec_ms=exec_ms,
                    active_cooldowns=active_cds,
                    annotated_fn=lambda: self.overlay.render(
                        frame, world, self.fsm.state, waypoint=self.fsm.navigator.current_waypoint, fps=fps_val
                    ),
                )

                annotated = None
                if self.show_overlay or chk.snapshot_requested or (
                    self.fsm.state == BotState.RECOVER_STUCK and prev_state != BotState.RECOVER_STUCK
                ):
                    annotated = self.overlay.render(
                        frame, world, self.fsm.state, waypoint=self.fsm.navigator.current_waypoint, fps=fps_val
                    )

                if chk.snapshot_requested:
                    self.recorder.save_async("manual", frame, world, self.fsm.state, annotated_bgr=annotated)
                elif self.fsm.state == BotState.RECOVER_STUCK and prev_state != BotState.RECOVER_STUCK:
                    self.recorder.save_async("stuck", frame, world, self.fsm.state, annotated_bgr=annotated)

                if self.show_overlay and annotated is not None:
                    key = self.overlay.show(annotated, wait_ms=1)
                    if key in (27, ord("q")):
                        self.fsm.emergency_stop(now, "overlay_quit")
                        break

                if self._transparent_overlay is not None and self._transparent_overlay.is_enabled:
                    self._transparent_overlay.update(
                        build_overlay_frame_data(
                            game_rect=(frame.origin_x, frame.origin_y, frame.width, frame.height),
                            world=world,
                            fsm_state=self.fsm.state,
                            config=self.config,
                            loop_hz=fps_val,
                            vision_ms=float(snapshot.timings_ms.get("total", 0.0)),
                            action_summary=act_summary or world.last_action_summary,
                            backend=self.pipeline.active_backend_desc,
                            input_mode=in_label,
                            game_focused=True,
                            travel_desc=self.fsm.travel_status_desc,
                            deaths_count=self.fsm.deaths_count,
                            repairs_count=self.fsm.repairs_count,
                            repair_rem_min=self.fsm.repair_remaining_minutes,
                        )
                    )

                prev_state = self.fsm.state
                tick_count += 1
                now_perf = time.perf_counter()
                if now_perf - fps_t0 >= 1.0:
                    fps_val = tick_count / (now_perf - fps_t0)
                    tick_count = 0
                    fps_t0 = now_perf

                sleep_s = tick_period - (now_perf - loop_t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            logger.info("Interrupted by Ctrl+C.")
        finally:
            self.executor.release_all()
            if hasattr(self.driver, "close"):
                self.driver.close()
            self.pipeline.close()
            capture.stop()
            self.overlay.close()
            if self._transparent_overlay is not None:
                if self._owns_transparent_overlay:
                    self._transparent_overlay.stop()
                else:
                    self._transparent_overlay.clear_and_hide()
            trace_summary = self.tracer.close()
            logger.info("%s", trace_summary)
            logger.info("Vikings Bot v2 stopped cleanly; all input keys/buttons released.")

    def run_replay(
        self,
        screenshots_dir: str | Path,
        name_filter: str = "Mortega-",
        delay_ms: int = 350,
        save_annotated_dir: str | Path | None = None,
        headless: bool = False,
    ) -> dict[str, int]:
        """Replay recorded screenshots through Pure OpenCV Perception + FSM (DryRun) + Overlay."""
        source = ReplayFrameSource(screenshots_dir, name_filter=name_filter)
        source.start()
        if not source.files:
            raise FileNotFoundError(f"No screenshots found in {screenshots_dir} matching {name_filter!r}")

        out_dir = Path(save_annotated_dir) if save_annotated_dir else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        self.fsm.resume(1000.0)
        stats = {
            "frames": 0,
            "coords_parsed": 0,
            "frames_with_mobs": 0,
            "frames_with_loot": 0,
        }

        try:
            while True:
                frame = source.latest_frame()
                if frame is None:
                    break
                snapshot = self.pipeline.analyze(frame)
                world = self.tracker.update(snapshot)
                intents = self.fsm.step(world)
                if intents:
                    self.executor.execute_many(intents, frame.timestamp)

                stats["frames"] += 1
                if snapshot.coords is not None:
                    stats["coords_parsed"] += 1
                if snapshot.mobs:
                    stats["frames_with_mobs"] += 1
                if snapshot.loot:
                    stats["frames_with_loot"] += 1

                annotated = self.overlay.render(
                    frame,
                    world,
                    self.fsm.state,
                    waypoint=self.fsm.navigator.current_waypoint,
                    fps=1000.0 / max(1.0, snapshot.timings_ms.get("total", 1.0)),
                )
                if out_dir is not None:
                    cv2.imwrite(str(out_dir / f"overlay_{frame.source_name}"), annotated)

                if not headless:
                    key = self.overlay.show(annotated, wait_ms=max(1, delay_ms))
                    if key in (27, ord("q")):
                        break
        finally:
            self.pipeline.close()
            self.overlay.close()
        return stats
