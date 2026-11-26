"""AutoHotkey v2 persistent stdin/stdout InputDriver supporting VM Foreground and Host Background."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from vikings_bot.capture.window import GameWindow, WindowTracker
from vikings_bot.input.driver import InputDriver

logger = logging.getLogger(__name__)

VALID_AHK_MODES = ("vm_foreground", "host_background")


def find_ahk_executable(configured_path: str | None = None) -> Path | None:
    """Locate AutoHotkey v2 executable (configured path, compiled bridge .exe, PATH, or Program Files)."""
    candidates: list[Path] = []
    if configured_path:
        candidates.append(Path(configured_path))
    for name in ("AutoHotkey64.exe", "AutoHotkey32.exe", "AutoHotkey.exe", "AutoHotkeyUX.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    candidates.extend(
        [
            Path(r"C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe"),
            Path(r"C:\Program Files\AutoHotkey\v2\AutoHotkey32.exe"),
            Path(r"C:\Program Files\AutoHotkey\AutoHotkey64.exe"),
            Path(r"C:\Program Files (x86)\AutoHotkey\v2\AutoHotkey64.exe"),
            Path(r"C:\Program Files (x86)\AutoHotkey\AutoHotkey.exe"),
        ]
    )
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    return None


def resolve_bridge_command(
    script_path: str | Path,
    ahk_exe: str | None = None,
    compiled_exe: str | Path | None = None,
) -> list[str] | None:
    """Build launch command for either a compiled Ahk2Exe bridge or AutoHotkey64.exe + .ahk script."""
    if compiled_exe:
        ce = Path(compiled_exe)
        if ce.exists() and ce.is_file():
            return [str(ce.resolve())]
    sp = Path(script_path)
    if sp.suffix.lower() == ".exe" and sp.exists():
        return [str(sp.resolve())]
    exe = find_ahk_executable(ahk_exe)
    if exe is None or not sp.exists():
        return None
    return [str(exe), str(sp.resolve())]


class AhkInputDriver(InputDriver):
    """Drives keyboard & mouse input through persistent AutoHotkey v2 worker (`vikings_input_bridge.ahk`).

    Supports two operating modes:
    - `vm_foreground`: absolute screen coordinates inside VMware Guest (host PC stays 100% free!)
    - `host_background`: client-relative `ControlClick` / `ControlSend` sent to game HWND without focus.
    """

    def __init__(
        self,
        *,
        ahk_mode: str = "vm_foreground",
        script_path: str | Path = "ahk/vikings_input_bridge.ahk",
        ahk_exe: str | None = None,
        compiled_exe: str | Path | None = None,
        window_tracker: WindowTracker | None = None,
        command_transport: Callable[[str], str] | None = None,
        fallback_driver: InputDriver | None = None,
    ) -> None:
        mode_norm = str(ahk_mode).strip().lower()
        if mode_norm not in VALID_AHK_MODES:
            mode_norm = "vm_foreground"
        self.ahk_mode = mode_norm
        self.script_path = Path(script_path)
        self.ahk_exe = ahk_exe
        self.compiled_exe = Path(compiled_exe) if compiled_exe else None
        self.window_tracker = window_tracker
        self._transport_override = command_transport
        self._fallback = fallback_driver

        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._started_once = False
        self._warned_missing = False
        self.history: list[str] = []

    @property
    def is_background_mode(self) -> bool:
        return self.ahk_mode == "host_background"

    def _current_window(self) -> GameWindow | None:
        if self.window_tracker is None:
            return None
        return self.window_tracker.find_window()

    def _target_hwnd(self) -> int:
        win = self._current_window()
        return int(win.hwnd) if win is not None else 0

    def _to_mode_xy(self, screen_x: int, screen_y: int) -> tuple[int, int, int]:
        """Return (hwnd, x, y) adjusted for vm_foreground (screen coords) or host_background (client coords)."""
        win = self._current_window()
        hwnd = int(win.hwnd) if win is not None else 0
        if self.ahk_mode == "host_background" and win is not None:
            return (hwnd, max(0, int(screen_x - win.left)), max(0, int(screen_y - win.top)))
        return (hwnd, int(screen_x), int(screen_y))

    def _ensure_process(self) -> subprocess.Popen[str] | None:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        cmd = resolve_bridge_command(self.script_path, self.ahk_exe, self.compiled_exe)
        if cmd is None:
            if not self._warned_missing:
                self._warned_missing = True
                logger.warning(
                    "AutoHotkey v2 executable not found (script=%s). Install AutoHotkey v2 or compile %s.",
                    self.script_path,
                    self.script_path.name,
                )
            return None
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=creationflags,
            )
            assert proc.stdout is not None
            ready = proc.stdout.readline().strip()
            if ready.startswith("READY"):
                logger.info("AHK v2 input bridge started (%s | mode=%s)", cmd[0], self.ahk_mode)
                self._proc = proc
                self._started_once = True
                return proc
            logger.warning("Unexpected AHK bridge startup reply: %r", ready)
            proc.terminate()
            return None
        except Exception as exc:
            logger.warning("Failed to start AHK v2 bridge: %s", exc)
            return None

    def send_raw(self, command_line: str) -> str:
        """Send one newline-delimited protocol line to the AHK v2 worker and return its reply."""
        cmd = command_line.strip()
        with self._lock:
            self.history.append(cmd)
            if self._transport_override is not None:
                return self._transport_override(cmd)
            proc = self._ensure_process()
            if proc is None or proc.stdin is None or proc.stdout is None:
                return "ERR|NO_AHK_PROCESS"
            try:
                proc.stdin.write(cmd + "\n")
                proc.stdin.flush()
                reply = proc.stdout.readline().strip()
                return reply or "ERR|EMPTY_REPLY"
            except Exception as exc:
                logger.warning("AHK bridge pipe error on %r: %s", cmd, exc)
                self._terminate_locked()
                return f"ERR|{exc}"

    def health_check(self) -> tuple[bool, str]:
        """Ping persistent AHK v2 worker to verify live round-trip communication."""
        reply = self.send_raw("PING")
        return (reply == "PONG", reply)

    def _run_or_fallback(self, cmd: str, fallback_fn: Callable[[], None] | None = None) -> None:
        reply = self.send_raw(cmd)
        if reply.startswith("ERR|NO_AHK_PROCESS") and self._fallback is not None and fallback_fn is not None:
            fallback_fn()

    def key_down(self, key: str) -> None:
        hwnd = self._target_hwnd()
        k = key.lower().strip()
        self._run_or_fallback(
            f"KEY_DOWN|{self.ahk_mode}|{hwnd}|{k}",
            (lambda: self._fallback.key_down(k)) if self._fallback is not None else None,
        )

    def key_up(self, key: str) -> None:
        hwnd = self._target_hwnd()
        k = key.lower().strip()
        self._run_or_fallback(
            f"KEY_UP|{self.ahk_mode}|{hwnd}|{k}",
            (lambda: self._fallback.key_up(k)) if self._fallback is not None else None,
        )

    def tap_key(self, key: str, hold_sec: float = 0.04) -> None:
        hwnd = self._target_hwnd()
        k = key.lower().strip()
        hold_ms = max(15, int(round(hold_sec * 1000.0)))
        self._run_or_fallback(
            f"KEY_TAP|{self.ahk_mode}|{hwnd}|{k}|{hold_ms}",
            (lambda: self._fallback.tap_key(k, hold_sec=hold_sec)) if self._fallback is not None else None,
        )

    def mouse_move_abs(self, x: int, y: int) -> None:
        hwnd, mx, my = self._to_mode_xy(x, y)
        self._run_or_fallback(
            f"MOVE|{self.ahk_mode}|{hwnd}|{mx}|{my}",
            (lambda: self._fallback.mouse_move_abs(x, y)) if self._fallback is not None else None,
        )

    def mouse_move_rel(
        self,
        dx: int,
        dy: int = 0,
        steps: int = 6,
        step_delay: float = 0.008,
        non_blocking: bool = False,
    ) -> None:
        # Pure Mouse Mode rotates camera via MMB drag; delegate relative move if ever requested
        if self._fallback is not None:
            self._fallback.mouse_move_rel(dx=dx, dy=dy, steps=steps, step_delay=step_delay, non_blocking=non_blocking)

    def mouse_click(self, x: int, y: int, button: str = "left", hold_sec: float = 0.04) -> None:
        hwnd, mx, my = self._to_mode_xy(x, y)
        btn = button.lower().strip()
        hold_ms = max(15, int(round(hold_sec * 1000.0)))
        self._run_or_fallback(
            f"CLICK|{self.ahk_mode}|{hwnd}|{mx}|{my}|{btn}|{hold_ms}",
            (lambda: self._fallback.mouse_click(x, y, button=btn, hold_sec=hold_sec)) if self._fallback is not None else None,
        )

    def combo_key_mouse_click(
        self,
        key: str = "space",
        button: str = "left",
        hold_sec: float = 0.08,
    ) -> None:
        self.key_down(key)
        time.sleep(max(0.015, hold_sec))
        self.key_up(key)

    def mouse_drag_rel(
        self,
        dx: int,
        dy: int = 0,
        button: str = "middle",
        steps: int = 14,
        step_delay: float = 0.014,
        start_x: int | None = None,
        start_y: int | None = None,
    ) -> None:
        win = self._current_window()
        if start_x is None or start_y is None:
            if win is not None:
                sx = win.left + win.width // 2
                sy = win.top + win.height // 2
            else:
                sx, sy = 512, 384
        else:
            sx, sy = int(start_x), int(start_y)
        hwnd, msx, msy = self._to_mode_xy(sx, sy)
        self._run_or_fallback(
            f"DRAG_MMB|{self.ahk_mode}|{hwnd}|{msx}|{msy}|{int(dx)}|{int(dy)}|{max(2, int(steps))}",
            (
                lambda: self._fallback.mouse_drag_rel(
                    dx=dx, dy=dy, button=button, steps=steps, step_delay=step_delay, start_x=sx, start_y=sy
                )
            )
            if self._fallback is not None
            else None,
        )

    def send_chat_command(self, command: str) -> None:
        hwnd = self._target_hwnd()
        clean = str(command).replace("|", "").replace("\n", "").strip()
        self._run_or_fallback(
            f"CHAT_CMD|{self.ahk_mode}|{hwnd}|{clean}",
            (lambda: self._fallback.send_chat_command(clean)) if self._fallback is not None else None,
        )

    def release_all(self) -> None:
        hwnd = self._target_hwnd()
        self._run_or_fallback(
            f"RELEASE_ALL|{hwnd}",
            (lambda: self._fallback.release_all()) if self._fallback is not None else None,
        )

    def _terminate_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._terminate_locked()
        if self._fallback is not None and hasattr(self._fallback, "close"):
            self._fallback.close()
