"""Win32 window discovery and foreground focus guard for Vikings Online."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

import psutil

user32 = ctypes.windll.user32

# Enable per-monitor DPI awareness so client rect matches physical 1920x1080 pixels
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    pid: int
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def monitor_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class WindowTracker:
    """Locates the Vikings Online game window (Minia.exe) and tracks foreground focus."""

    def __init__(
        self,
        process_name: str = "Minia.exe",
        title_substring: str = "Vikings",
        process_aliases: list[str] | None = None,
        title_aliases: list[str] | None = None,
    ) -> None:
        p_set = {process_name.lower()}
        if process_aliases:
            p_set.update(p.lower() for p in process_aliases)
        t_set = {title_substring.lower()} if title_substring else set()
        if title_aliases:
            t_set.update(t.lower() for t in title_aliases if t)
        self.process_names = p_set
        self.title_substrings = t_set
        self._cached_hwnd: int | None = None

    def _find_target_pids(self) -> set[int]:
        pids: set[int] = set()
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name in self.process_names:
                    pids.add(int(proc.info["pid"]))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    @staticmethod
    def get_foreground_info() -> tuple[int, int, str]:
        fg = int(user32.GetForegroundWindow())
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
        return (fg, int(pid.value), WindowTracker._get_window_title(fg))

    @staticmethod
    def _get_window_title(hwnd: int) -> str:
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    @staticmethod
    def _get_client_bounds(hwnd: int) -> tuple[int, int, int, int] | None:
        rect = _RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < 200 or height < 200:
            return None
        origin = _POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        return (int(origin.x), int(origin.y), width, height)

    def find_window(self) -> GameWindow | None:
        # Fast path: reuse cached HWND if still valid and visible
        if self._cached_hwnd and user32.IsWindow(self._cached_hwnd) and user32.IsWindowVisible(self._cached_hwnd):
            bounds = self._get_client_bounds(self._cached_hwnd)
            if bounds is not None:
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(self._cached_hwnd, ctypes.byref(pid))
                return GameWindow(
                    hwnd=self._cached_hwnd,
                    pid=int(pid.value),
                    title=self._get_window_title(self._cached_hwnd),
                    left=bounds[0],
                    top=bounds[1],
                    width=bounds[2],
                    height=bounds[3],
                )

        target_pids = self._find_target_pids()
        matches: list[GameWindow] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def enum_cb(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            title = self._get_window_title(hwnd)
            title_low = title.lower()
            pid_match = int(pid.value) in target_pids
            title_match = any(sub in title_low for sub in self.title_substrings)
            if not (pid_match or title_match):
                return True
            bounds = self._get_client_bounds(hwnd)
            if bounds is None:
                return True
            matches.append(
                GameWindow(
                    hwnd=int(hwnd),
                    pid=int(pid.value),
                    title=title,
                    left=bounds[0],
                    top=bounds[1],
                    width=bounds[2],
                    height=bounds[3],
                )
            )
            return True

        user32.EnumWindows(enum_cb, 0)
        if not matches:
            self._cached_hwnd = None
            return None

        # Prefer PID match with largest client area
        matches.sort(key=lambda w: (w.pid in target_pids, w.width * w.height), reverse=True)
        chosen = matches[0]
        self._cached_hwnd = chosen.hwnd
        return chosen

    def is_focused(self, window: GameWindow | None = None) -> bool:
        win = window or self.find_window()
        if win is None:
            return False
        fg = int(user32.GetForegroundWindow())
        if fg == win.hwnd:
            return True
        fg_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(fg, ctypes.byref(fg_pid))
        return int(fg_pid.value) == win.pid
