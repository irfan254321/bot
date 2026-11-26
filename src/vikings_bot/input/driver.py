"""Standard Windows user-space input driver (ctypes SendInput) and DryRun driver."""

from __future__ import annotations

import ctypes
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from ctypes import wintypes

from vikings_bot.input.humanizer import ease_mouse_deltas

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32

ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Windows Virtual Key codes (sent alongside scan codes so both DirectInput & Win32/RYL read WASD/E/Space)
VK_KEY_MAP: dict[str, int] = {
    "esc": 0x1B,
    "0": 0x30,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "5": 0x35,
    "6": 0x36,
    "7": 0x37,
    "8": 0x38,
    "9": 0x39,
    "tab": 0x09,
    "enter": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "space": 0x20,
    "a": 0x41,
    "b": 0x42,
    "c": 0x43,
    "d": 0x44,
    "e": 0x45,
    "f": 0x46,
    "g": 0x47,
    "h": 0x48,
    "i": 0x49,
    "j": 0x4A,
    "k": 0x4B,
    "l": 0x4C,
    "m": 0x4D,
    "n": 0x4E,
    "o": 0x4F,
    "p": 0x50,
    "q": 0x51,
    "r": 0x52,
    "s": 0x53,
    "t": 0x54,
    "u": 0x55,
    "v": 0x56,
    "w": 0x57,
    "x": 0x58,
    "y": 0x59,
    "z": 0x5A,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
    "/": 0xBF,
    "slash": 0xBF,
}

# DirectInput-compatible US keyboard hardware scan codes
SCAN_CODES: dict[str, int] = {
    "esc": 0x01,
    "1": 0x02,
    "2": 0x03,
    "3": 0x04,
    "4": 0x05,
    "5": 0x06,
    "6": 0x07,
    "7": 0x08,
    "8": 0x09,
    "9": 0x0A,
    "0": 0x0B,
    "tab": 0x0F,
    "q": 0x10,
    "w": 0x11,
    "e": 0x12,
    "r": 0x13,
    "t": 0x14,
    "y": 0x15,
    "u": 0x16,
    "i": 0x17,
    "o": 0x18,
    "p": 0x19,
    "enter": 0x1C,
    "ctrl": 0x1D,
    "a": 0x1E,
    "s": 0x1F,
    "d": 0x20,
    "f": 0x21,
    "g": 0x22,
    "h": 0x23,
    "j": 0x24,
    "k": 0x25,
    "l": 0x26,
    "z": 0x2C,
    "x": 0x2D,
    "c": 0x2E,
    "v": 0x2F,
    "b": 0x30,
    "n": 0x31,
    "m": 0x32,
    "shift": 0x2A,
    "alt": 0x38,
    "space": 0x39,
    "f1": 0x3B,
    "f2": 0x3C,
    "f3": 0x3D,
    "f4": 0x3E,
    "f5": 0x3F,
    "f6": 0x40,
    "f7": 0x41,
    "f8": 0x42,
    "f9": 0x43,
    "f10": 0x44,
    "f11": 0x57,
    "f12": 0x58,
    "/": 0x35,
    "slash": 0x35,
}


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]


class InputDriver(ABC):
    @abstractmethod
    def key_down(self, key: str) -> None: ...

    @abstractmethod
    def key_up(self, key: str) -> None: ...

    @abstractmethod
    def tap_key(self, key: str, hold_sec: float = 0.04) -> None: ...

    @abstractmethod
    def mouse_move_abs(self, x: int, y: int) -> None: ...

    @abstractmethod
    def mouse_move_rel(
        self,
        dx: int,
        dy: int = 0,
        steps: int = 6,
        step_delay: float = 0.008,
        non_blocking: bool = False,
    ) -> None: ...

    @abstractmethod
    def mouse_click(self, x: int, y: int, button: str = "left", hold_sec: float = 0.04) -> None: ...

    @abstractmethod
    def combo_key_mouse_click(
        self,
        key: str = "space",
        button: str = "left",
        hold_sec: float = 0.08,
    ) -> None: ...

    @abstractmethod
    def mouse_drag_rel(
        self,
        dx: int,
        dy: int = 0,
        button: str = "middle",
        steps: int = 14,
        step_delay: float = 0.014,
        start_x: int | None = None,
        start_y: int | None = None,
    ) -> None: ...

    @abstractmethod
    def release_all(self) -> None: ...

    def set_e_spam(self, enabled: bool) -> None:
        """Enable or disable continuous background 'E' pickup spam."""
        return None

    def send_chat_command(self, command: str) -> None:
        """Open RYL chat (Enter), type command characters (e.g. '/kill'), and submit (Enter)."""
        self.tap_key("enter", hold_sec=0.045)
        time.sleep(0.12)
        for ch in command:
            key = "space" if ch == " " else ch.lower()
            self.tap_key(key, hold_sec=0.035)
            time.sleep(0.035)
        time.sleep(0.08)
        self.tap_key("enter", hold_sec=0.045)


class Win32InputDriver(InputDriver):
    """Standard Windows SendInput driver with scan codes, 125Hz async smooth camera, and VM absolute mode."""

    def __init__(self, vm_absolute_mouse: bool = False) -> None:
        self.vm_absolute_mouse = bool(vm_absolute_mouse)
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        self._last_fail_warn_ts: float = 0.0
        self._mouse_lock = threading.Lock()
        self._mouse_queue: deque[tuple[int, int, float]] = deque()
        self._mouse_wake = threading.Event()
        self._mouse_stop = threading.Event()
        self._e_spam_enabled = threading.Event()
        self._e_spam_chat_lock = threading.Event()
        self._mouse_worker = threading.Thread(
            target=self._smooth_mouse_loop,
            name="vikings-smooth-mouse",
            daemon=True,
        )
        self._mouse_worker.start()
        self._e_spam_worker = threading.Thread(
            target=self._e_spam_loop,
            name="vikings-win32-e-spam",
            daemon=True,
        )
        self._e_spam_worker.start()

    def _e_spam_loop(self) -> None:
        """Win32 fallback continuous background 'E' pickup spam (90ms interval) when AHK bridge is unavailable."""
        while not self._mouse_stop.is_set():
            if self._e_spam_enabled.is_set() and not self._e_spam_chat_lock.is_set():
                self.tap_key("e", hold_sec=0.025)
                time.sleep(0.065)
            else:
                time.sleep(0.04)

    def set_e_spam(self, enabled: bool) -> None:
        if enabled:
            self._e_spam_enabled.set()
        else:
            self._e_spam_enabled.clear()

    def send_chat_command(self, command: str) -> None:
        self._e_spam_chat_lock.set()
        try:
            time.sleep(0.06)
            super().send_chat_command(command)
            time.sleep(0.08)
        finally:
            self._e_spam_chat_lock.clear()

    def _smooth_mouse_loop(self) -> None:
        """Executes S-curve relative camera deltas asynchronously so perception stays locked at 30 FPS."""
        while not self._mouse_stop.is_set():
            item: tuple[int, int, float] | None = None
            with self._mouse_lock:
                if self._mouse_queue:
                    item = self._mouse_queue.popleft()
                else:
                    self._mouse_wake.clear()
            if item is None:
                self._mouse_wake.wait(timeout=0.05)
                continue
            sx, sy, delay_s = item
            if sx != 0 or sy != 0:
                inp = _INPUT(
                    type=INPUT_MOUSE,
                    union=_INPUT_UNION(
                        mi=_MOUSEINPUT(dx=sx, dy=sy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0)
                    ),
                )
                self._send(inp)
            if delay_s > 0.0:
                time.sleep(delay_s)

    def cancel_async_mouse(self) -> None:
        """Immediately discard any queued relative camera deltas (called on click, drag, pause, or stop)."""
        with self._mouse_lock:
            self._mouse_queue.clear()
            self._mouse_wake.clear()

    def close(self) -> None:
        self._e_spam_enabled.clear()
        self.cancel_async_mouse()
        self._mouse_stop.set()
        self._mouse_wake.set()

    @staticmethod
    def get_cursor_pos() -> tuple[int, int] | None:
        """Return current Win32 desktop cursor (x, y) or None if unavailable."""
        pt = _POINT()
        if user32.GetCursorPos(ctypes.byref(pt)):
            return (int(pt.x), int(pt.y))
        return None

    def _send(self, inp: _INPUT) -> bool:
        return self._send_batch([inp])

    def _send_batch(self, inputs: list[_INPUT]) -> bool:
        if not inputs:
            return True
        arr = (_INPUT * len(inputs))(*inputs)
        sent = int(user32.SendInput(len(inputs), arr, ctypes.sizeof(_INPUT)))
        if sent == 0:
            now = time.time()
            if now - self._last_fail_warn_ts >= 5.0:
                self._last_fail_warn_ts = now
                logger.warning(
                    "Win32 SendInput returned 0 (blocked by OS/UIPI or anti-cheat). "
                    "Make sure PowerShell is running as Administrator if the game is elevated."
                )
            return False
        return True

    def _make_key_input(self, key: str, down: bool) -> _INPUT:
        k = key.lower().strip()
        sc = self._scan_code(k)
        vk = VK_KEY_MAP.get(k, 0)
        # Populate BOTH wVk and wScan (without KEYEVENTF_SCANCODE masking wVk) so both
        # DirectInput and Win32 GetAsyncKeyState/WM_KEYDOWN in RYL Keyboard Mode register cleanly.
        flags = 0 if down else KEYEVENTF_KEYUP
        return _INPUT(
            type=INPUT_KEYBOARD,
            union=_INPUT_UNION(ki=_KEYBDINPUT(wVk=vk, wScan=sc, dwFlags=flags, time=0, dwExtraInfo=0)),
        )

    @staticmethod
    def _make_mouse_btn_input(button: str, down: bool) -> _INPUT:
        b = button.lower()
        if b == "left":
            flag = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
        elif b == "right":
            flag = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP
        elif b == "middle":
            flag = MOUSEEVENTF_MIDDLEDOWN if down else MOUSEEVENTF_MIDDLEUP
        else:
            raise ValueError(f"Unsupported mouse button: {button!r}")
        return _INPUT(
            type=INPUT_MOUSE,
            union=_INPUT_UNION(mi=_MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flag, time=0, dwExtraInfo=0)),
        )

    @staticmethod
    def is_admin() -> bool:
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

    @staticmethod
    def _scan_code(key: str) -> int:
        k = key.lower().strip()
        if k not in SCAN_CODES:
            raise ValueError(f"Unsupported key name: {key!r}")
        return SCAN_CODES[k]

    @staticmethod
    def _to_abs_coords(x: int, y: int) -> tuple[int, int]:
        vx = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        vy = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        vw = max(1, int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)))
        vh = max(1, int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)))
        nx = int(round(((x - vx) * 65535) / max(1, vw - 1)))
        ny = int(round(((y - vy) * 65535) / max(1, vh - 1)))
        return (max(0, min(65535, nx)), max(0, min(65535, ny)))

    def key_down(self, key: str) -> None:
        k = key.lower().strip()
        self._send(self._make_key_input(k, down=True))
        self._held_keys.add(k)

    def key_up(self, key: str) -> None:
        k = key.lower().strip()
        self._send(self._make_key_input(k, down=False))
        self._held_keys.discard(k)

    def tap_key(self, key: str, hold_sec: float = 0.04) -> None:
        self.key_down(key)
        time.sleep(max(0.01, hold_sec))
        self.key_up(key)

    def _send_abs_move(self, x: int, y: int) -> None:
        nx, ny = self._to_abs_coords(int(x), int(y))
        flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        inp = _INPUT(
            type=INPUT_MOUSE,
            union=_INPUT_UNION(mi=_MOUSEINPUT(dx=nx, dy=ny, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)),
        )
        self._send(inp)

    def mouse_move_abs(self, x: int, y: int) -> None:
        """Move cursor smoothly & rapidly (~20-30ms) to exact absolute screen (x, y) with zero OS acceleration drift.

        Uses a crisp 4-6 step cubic S-curve of PURE MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK events
        followed by exact SetCursorPos settle on BOTH bare-metal PC and VMware — eliminating the old
        130-170ms relative-acceleration glide that caused clicks to land 100-180px behind walking monsters!
        """
        self.cancel_async_mouse()
        tx, ty = int(x), int(y)
        pt = _POINT()
        has_pt = bool(user32.GetCursorPos(ctypes.byref(pt)))

        if has_pt:
            cx, cy = int(pt.x), int(pt.y)
            dx = tx - cx
            dy = ty - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= 4.0:
                if self.vm_absolute_mouse:
                    steps = max(4, min(7, int(round(dist / 130.0)) + 3))
                    step_delay = 0.0034
                else:
                    steps = max(4, min(6, int(round(dist / 160.0)) + 3))
                    step_delay = 0.0032
                deltas = ease_mouse_deltas(dx=dx, dy=dy, steps=steps, add_micro_arc=False)
                cur_x, cur_y = cx, cy
                n = len(deltas)
                for idx, (sx, sy) in enumerate(deltas):
                    cur_x += sx
                    cur_y += sy
                    self._send_abs_move(cur_x, cur_y)
                    if n > 1 and idx < n - 1:
                        time.sleep(step_delay * random.uniform(0.84, 1.16))

        self._send_abs_move(tx, ty)
        user32.SetCursorPos(tx, ty)

    def mouse_move_rel(
        self,
        dx: int,
        dy: int = 0,
        steps: int = 6,
        step_delay: float = 0.006,
        non_blocking: bool = False,
    ) -> None:
        """Smooth cubic S-curve relative mouse motion with STRICT zero vertical pitch drift (dy=0 stays sy=0).

        When non_blocking=True, enqueues the S-curve deltas onto the dedicated smooth-mouse worker thread
        (replacing any stale tail) so the 30 FPS perception loop NEVER stalls during camera rotation!
        """
        deltas = ease_mouse_deltas(dx=dx, dy=dy, steps=max(1, steps), add_micro_arc=False)
        n = len(deltas)
        if non_blocking:
            scheduled: list[tuple[int, int, float]] = []
            for idx, (sx, sy) in enumerate(deltas):
                dly = (step_delay * random.uniform(0.82, 1.18)) if (n > 1 and idx < n - 1 and step_delay > 0) else 0.0
                scheduled.append((sx, sy, dly))
            with self._mouse_lock:
                self._mouse_queue.clear()
                self._mouse_queue.extend(scheduled)
                self._mouse_wake.set()
            return

        for idx, (sx, sy) in enumerate(deltas):
            if sx != 0 or sy != 0:
                inp = _INPUT(
                    type=INPUT_MOUSE,
                    union=_INPUT_UNION(
                        mi=_MOUSEINPUT(dx=sx, dy=sy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0)
                    ),
                )
                self._send(inp)
            if n > 1 and idx < n - 1 and step_delay > 0:
                time.sleep(step_delay * random.uniform(0.80, 1.22))

    def _mouse_button(self, button: str, down: bool) -> None:
        b = button.lower()
        if b == "left":
            flag = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
        elif b == "right":
            flag = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP
        elif b == "middle":
            flag = MOUSEEVENTF_MIDDLEDOWN if down else MOUSEEVENTF_MIDDLEUP
        else:
            raise ValueError(f"Unsupported mouse button: {button!r}")
        inp = _INPUT(
            type=INPUT_MOUSE,
            union=_INPUT_UNION(mi=_MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flag, time=0, dwExtraInfo=0)),
        )
        self._send(inp)
        if down:
            self._held_buttons.add(b)
        else:
            self._held_buttons.discard(b)

    def mouse_click(self, x: int, y: int, button: str = "left", hold_sec: float = 0.030) -> None:
        self.mouse_move_abs(x, y)
        time.sleep(random.uniform(0.008, 0.015))
        self._mouse_button(button, down=True)
        time.sleep(max(0.020, hold_sec * random.uniform(0.85, 1.18)))
        self._mouse_button(button, down=False)

    def combo_key_mouse_click(
        self,
        key: str = "space",
        button: str = "left",
        hold_sec: float = 0.032,
    ) -> None:
        """Press key (Space) + mouse button (Left Click) simultaneously with organic hold variance."""
        k = key.lower().strip()
        b = button.lower().strip()
        down_batch = [self._make_key_input(k, down=True), self._make_mouse_btn_input(b, down=True)]
        up_batch = [self._make_mouse_btn_input(b, down=False), self._make_key_input(k, down=False)]
        self._held_keys.add(k)
        self._held_buttons.add(b)
        try:
            self._send_batch(down_batch)
            time.sleep(max(0.022, hold_sec * random.uniform(0.85, 1.18)))
        finally:
            self._send_batch(up_batch)
            self._held_buttons.discard(b)
            self._held_keys.discard(k)

    def mouse_drag_rel(
        self,
        dx: int,
        dy: int = 0,
        button: str = "middle",
        steps: int = 12,
        step_delay: float = 0.008,
        start_x: int | None = None,
        start_y: int | None = None,
    ) -> None:
        """Smooth S-curve Middle-Mouse camera drag with ZERO vertical pitch drift (bare-metal & VMware)."""
        self.cancel_async_mouse()
        steps = max(8, steps)
        if start_x is not None and start_y is not None:
            self.mouse_move_abs(start_x, start_y)
            time.sleep(random.uniform(0.022, 0.036))
        self._mouse_button(button, down=True)
        time.sleep(random.uniform(0.020, 0.034))
        try:
            deltas = ease_mouse_deltas(dx=dx, dy=dy, steps=steps, add_micro_arc=False)
            # Always emit pure relative MOUSEEVENTF_MOVE (with exact dy=0 for horizontal sweeps) while MMB is held:
            # RYL/Vikings reads DirectInput relative deltas during MMB rotation, whereas MOUSEEVENTF_ABSOLUTE
            # under VMware vmmouse caused repeated vertical pitch drift slamming the camera straight up/down!
            for step_x, step_y in deltas:
                if step_x != 0 or step_y != 0:
                    inp = _INPUT(
                        type=INPUT_MOUSE,
                        union=_INPUT_UNION(
                            mi=_MOUSEINPUT(
                                dx=step_x, dy=step_y, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0
                            )
                        ),
                    )
                    self._send(inp)
                time.sleep(step_delay * random.uniform(0.80, 1.22))
            time.sleep(random.uniform(0.018, 0.030))
        finally:
            self._mouse_button(button, down=False)

    def release_all(self) -> None:
        self.cancel_async_mouse()
        for key in list(self._held_keys):
            try:
                self.key_up(key)
            except Exception:
                self._held_keys.discard(key)
        for btn in list(self._held_buttons):
            try:
                self._mouse_button(btn, down=False)
            except Exception:
                self._held_buttons.discard(btn)


class DryRunInputDriver(InputDriver):
    """Logs every action without sending any OS keyboard or mouse events."""

    def __init__(self) -> None:
        self.history: list[str] = []

    def key_down(self, key: str) -> None:
        msg = f"key_down({key})"
        self.history.append(msg)
        logger.debug("[DRY-RUN] %s", msg)

    def key_up(self, key: str) -> None:
        msg = f"key_up({key})"
        self.history.append(msg)
        logger.debug("[DRY-RUN] %s", msg)

    def tap_key(self, key: str, hold_sec: float = 0.04) -> None:
        msg = f"tap_key({key})"
        self.history.append(msg)
        logger.info("[DRY-RUN] %s", msg)

    def mouse_move_abs(self, x: int, y: int) -> None:
        msg = f"mouse_move_abs({x}, {y})"
        self.history.append(msg)
        logger.debug("[DRY-RUN] %s", msg)

    def mouse_move_rel(
        self,
        dx: int,
        dy: int = 0,
        steps: int = 6,
        step_delay: float = 0.008,
        non_blocking: bool = False,
    ) -> None:
        msg = f"mouse_move_rel(dx={dx}, dy={dy})"
        self.history.append(msg)
        logger.info("[DRY-RUN] %s", msg)

    def mouse_click(self, x: int, y: int, button: str = "left", hold_sec: float = 0.04) -> None:
        msg = f"mouse_click({button}, {x}, {y})"
        self.history.append(msg)
        logger.info("[DRY-RUN] %s", msg)

    def combo_key_mouse_click(
        self,
        key: str = "space",
        button: str = "left",
        hold_sec: float = 0.08,
    ) -> None:
        msg = f"combo_key_mouse_click({key}+{button})"
        self.history.append(msg)
        logger.info("[DRY-RUN] %s", msg)

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
        msg = f"mouse_drag_rel({button}, dx={dx}, dy={dy})"
        self.history.append(msg)
        logger.info("[DRY-RUN] %s", msg)

    def release_all(self) -> None:
        self.history.append("release_all()")
        logger.debug("[DRY-RUN] release_all()")

    def set_e_spam(self, enabled: bool) -> None:
        flag = 1 if enabled else 0
        self.history.append(f"set_e_spam({flag})")
        logger.debug("[DRY-RUN] set_e_spam(%d)", flag)

    def send_chat_command(self, command: str) -> None:
        msg = f"send_chat_command({command})"
        self.history.append(msg)
        logger.info("[DRY-RUN] %s", msg)


def create_input_driver(
    config: object,
    *,
    live_input: bool = False,
    window_tracker: object | None = None,
) -> InputDriver:
    """Create the configured InputDriver ('ahk' -> AHK v2 bridge with Win32 fallback, 'win32', or DryRun)."""
    if not live_input:
        return DryRunInputDriver()

    controls = getattr(config, "controls", {}) or {}
    backend = str(controls.get("input_backend", "ahk")).strip().lower()
    vm_abs = bool(controls.get("vm_absolute_mouse", True))
    win32_drv = Win32InputDriver(vm_absolute_mouse=vm_abs)

    if backend == "ahk":
        from vikings_bot.input.ahk_driver import AhkInputDriver

        ahk_mode = str(controls.get("ahk_mode", "vm_foreground")).strip().lower()
        script_rel = str(controls.get("ahk_script_path", "ahk/vikings_input_bridge.ahk"))
        script_path = config.resolve_path(script_rel) if hasattr(config, "resolve_path") else script_rel
        ahk_exe = controls.get("ahk_exe_path") or None
        compiled_rel = controls.get("ahk_compiled_exe") or None
        compiled_exe = (
            config.resolve_path(compiled_rel)
            if (compiled_rel and hasattr(config, "resolve_path"))
            else compiled_rel
        )
        allow_fallback = bool(controls.get("ahk_win32_fallback", True))
        return AhkInputDriver(
            ahk_mode=ahk_mode,
            script_path=script_path,
            ahk_exe=str(ahk_exe) if ahk_exe else None,
            compiled_exe=compiled_exe,
            window_tracker=window_tracker,  # type: ignore[arg-type]
            fallback_driver=win32_drv if allow_fallback else None,
        )

    return win32_drv
