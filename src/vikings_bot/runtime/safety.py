"""Global hotkeys (F12 kill, F9 pause/resume, F10 snapshot), focus guard, death guard, audio cues."""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from dataclasses import dataclass

from vikings_bot.capture.window import WindowTracker
from vikings_bot.config import BotConfig
from vikings_bot.state.models import WorldState

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32

VK_CODES: dict[str, int] = {
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
    "esc": 0x1B,
    "pause": 0x13,
    "end": 0x23,
}


class AudioCuePlayer:
    """Non-blocking short beep cues for single-monitor Fullscreen status awareness."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def _play_sequence(self, tones: list[tuple[int, int]]) -> None:
        try:
            import winsound

            for freq, dur_ms in tones:
                winsound.Beep(freq, dur_ms)
                time.sleep(0.03)
        except Exception:
            pass

    def play(self, event: str) -> None:
        if not self.enabled:
            return
        patterns: dict[str, list[tuple[int, int]]] = {
            "resume": [(660, 80), (990, 110)],
            "pause": [(880, 80), (520, 110)],
            "kill": [(1040, 90), (780, 90), (440, 180)],
            "alert": [(600, 100), (600, 100)],
            "snapshot": [(1200, 60)],
            "overlay_on": [(780, 50)],
            "overlay_off": [(520, 50)],
        }
        seq = patterns.get(event)
        if not seq:
            return
        threading.Thread(target=self._play_sequence, args=(seq,), daemon=True).start()


class HotkeyPoller:
    """Edge-triggered Windows global hotkey checker via GetAsyncKeyState with bit-0 short-tap latch."""

    def __init__(self, key_names: dict[str, str], debounce_sec: float = 0.25) -> None:
        self._keys: dict[str, int] = {}
        self._was_down: dict[str, bool] = {}
        self._last_fire_ts: dict[str, float] = {}
        self._debounce_sec = max(0.05, float(debounce_sec))
        for action, key_name in key_names.items():
            vk = VK_CODES.get(key_name.lower().strip())
            if vk is None:
                raise ValueError(f"Unsupported hotkey {key_name!r} for {action!r}")
            self._keys[action] = vk
            self._was_down[action] = False
            self._last_fire_ts[action] = -999.0
            # Prime & clear any stale bit-0 latched press from before engine started
            try:
                user32.GetAsyncKeyState(vk)
            except Exception:
                pass

    def poll(self) -> set[str]:
        now = time.monotonic()
        triggered: set[str] = set()
        for action, vk in self._keys.items():
            state = int(user32.GetAsyncKeyState(vk))
            is_down = bool(state & 0x8000)
            was_pressed_since_last = bool(state & 0x0001)
            # Fire on rising edge OR latched short tap (bit 0) so a 30ms F9 tap is NEVER missed!
            if ((is_down and not self._was_down[action]) or (was_pressed_since_last and not self._was_down[action])) and (
                (now - self._last_fire_ts[action]) >= self._debounce_sec
            ):
                triggered.add(action)
                self._last_fire_ts[action] = now
            self._was_down[action] = is_down
        return triggered


@dataclass(frozen=True)
class SafetyCheckResult:
    kill_triggered: bool = False
    toggle_pause: bool = False
    snapshot_requested: bool = False
    dataset_capture_requested: bool = False
    force_repair_requested: bool = False
    force_pause_reason: str | None = None
    toggle_overlay: bool = False


class SafetyMonitor:
    """Combines hotkeys, window focus guard, death guard, and max-stuck protection."""

    def __init__(
        self,
        config: BotConfig,
        window_tracker: WindowTracker | None = None,
        enable_hotkeys: bool = True,
    ) -> None:
        s = config.safety
        w = config.window
        rt = config.respawn_travel
        self.require_focus = bool(w.get("require_focus", True))
        self.death_zero_hp_frames = int(s.get("death_zero_hp_frames", 12))
        self.max_consecutive_stuck = int(s.get("max_consecutive_stuck", 6))
        self.auto_respawn_enabled = bool(rt.get("enabled", True))
        self.window_tracker = window_tracker
        self.audio = AudioCuePlayer(enabled=bool(s.get("audio_cues", True)))
        self.hotkeys: HotkeyPoller | None = None
        if enable_hotkeys:
            self.hotkeys = HotkeyPoller(
                {
                    "kill": str(s.get("kill_switch", "F10")),
                    "pause": str(s.get("pause_toggle", "F9")),
                    "dataset_capture": str(s.get("dataset_capture_key", "F8")),
                    "force_repair": str(s.get("force_repair_key", "F7")),
                    "snapshot": str(s.get("snapshot_key", "F11")),
                    "overlay": str(s.get("overlay_toggle", "F6")),
                }
            )

    def check(
        self,
        world: WorldState | None,
        is_running_active: bool,
        in_respawn_or_travel: bool = False,
    ) -> SafetyCheckResult:
        triggered = self.hotkeys.poll() if self.hotkeys is not None else set()

        if "kill" in triggered:
            self.audio.play("kill")
            return SafetyCheckResult(kill_triggered=True)

        snap = "snapshot" in triggered
        ds_cap = "dataset_capture" in triggered
        force_rep = "force_repair" in triggered
        if snap or ds_cap:
            self.audio.play("snapshot")
        if force_rep:
            self.audio.play("alert")

        toggle = "pause" in triggered
        toggle_overlay = "overlay" in triggered
        force_pause: str | None = None

        if is_running_active:
            if self.require_focus and self.window_tracker is not None and not self.window_tracker.is_focused():
                force_pause = "game_window_lost_focus"
            elif (
                not self.auto_respawn_enabled
                and world is not None
                and world.is_dead
            ):
                force_pause = "player_dead_zero_hp"
            elif (
                not in_respawn_or_travel
                and world is not None
                and world.consecutive_stuck >= self.max_consecutive_stuck
            ):
                force_pause = "max_consecutive_stuck_exceeded"

        if force_pause is not None:
            self.audio.play("alert")

        return SafetyCheckResult(
            kill_triggered=False,
            toggle_pause=toggle,
            snapshot_requested=snap,
            dataset_capture_requested=ds_cap,
            force_repair_requested=force_rep,
            force_pause_reason=force_pause,
            toggle_overlay=toggle_overlay,
        )
